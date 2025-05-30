#!/usr/bin/env python3
"""
Database initialization script for LeMCS
Sets up PostgreSQL with pgvector extension and creates all tables
"""
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import Base
from config.settings import settings


async def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    # Parse database URL to get connection info
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql+asyncpg://"):
        # Extract connection details for creating database
        url_parts = db_url.replace("postgresql+asyncpg://", "").split("/")
        base_url = url_parts[0]
        db_name = url_parts[1] if len(url_parts) > 1 else "lemcs_db"
        
        # Connect to postgres database to create our database
        postgres_url = f"postgresql://{base_url}/postgres"
        
        try:
            conn = await asyncpg.connect(postgres_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                print(f"✅ Created database: {db_name}")
            else:
                print(f"📍 Database already exists: {db_name}")
                
            await conn.close()
        except Exception as e:
            print(f"⚠️  Could not create database: {e}")
            print("📍 Assuming database exists and continuing...")


async def setup_pgvector_extension():
    """Set up the pgvector extension"""
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        try:
            # Create pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("✅ pgvector extension enabled")
            
            # Create additional useful extensions
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            print("✅ pg_trgm extension enabled (for text search)")
            
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
            print("✅ btree_gin extension enabled (for indexing)")
            
        except Exception as e:
            print(f"⚠️  Error setting up extensions: {e}")
            raise
    
    await engine.dispose()


async def create_tables():
    """Create all database tables"""
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        try:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            print("✅ All database tables created successfully")
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    await engine.dispose()


async def create_indexes():
    """Create optimized indexes for performance"""
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        try:
            # Vector similarity indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector 
                ON document_embeddings USING hnsw (embedding vector_cosine_ops)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_citation_embeddings_vector 
                ON citation_embeddings USING hnsw (embedding vector_cosine_ops)
            """))
            
            # Text search indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_documents_extracted_text_gin 
                ON documents USING gin (to_tsvector('english', extracted_text))
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_provisions_text_gin 
                ON document_provisions USING gin (to_tsvector('english', provision_text))
            """))
            
            # Performance indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_documents_status 
                ON documents (status)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_documents_type 
                ON documents (document_type)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_workflows_status 
                ON agent_workflows (status)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_consolidation_jobs_status 
                ON consolidation_jobs (status)
            """))
            
            print("✅ Performance indexes created successfully")
            
        except Exception as e:
            print(f"⚠️  Error creating indexes: {e}")
            # Don't raise - indexes are optimization, not critical
    
    await engine.dispose()


async def seed_legal_theories():
    """Seed the database with common legal theories"""
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        try:
            # Insert common legal theories for residential lease analysis
            theories = [
                ("Implied Warranty of Habitability", "Legal doctrine requiring landlords to maintain rental properties in habitable condition"),
                ("Truth in Renting Act", "Statutes requiring disclosure of material facts about rental properties"),
                ("Unconscionability", "Legal doctrine preventing enforcement of contracts that are extremely unfair"),
                ("Constructive Eviction", "Doctrine allowing tenants to terminate lease when landlord's actions make property uninhabitable"),
                ("Quiet Enjoyment", "Tenant's right to use rental property without unreasonable interference from landlord"),
                ("Security Deposit Law", "Statutory protections governing collection and return of tenant security deposits"),
                ("Rent Control/Stabilization", "Laws limiting rent increases and providing tenant protections"),
                ("Fair Housing Act", "Federal law prohibiting discrimination in housing based on protected characteristics"),
                ("Retaliatory Eviction", "Prohibition against evicting tenants for exercising legal rights"),
                ("Breach of Lease", "Violation of lease terms by either landlord or tenant")
            ]
            
            for name, description in theories:
                await conn.execute(text("""
                    INSERT INTO legal_theories (id, name, description, created_at)
                    VALUES (gen_random_uuid(), :name, :description, NOW())
                    ON CONFLICT (name) DO NOTHING
                """), {"name": name, "description": description})
            
            print("✅ Legal theories seeded successfully")
            
        except Exception as e:
            print(f"⚠️  Error seeding legal theories: {e}")
    
    await engine.dispose()


async def verify_setup():
    """Verify the database setup is working correctly"""
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        try:
            # Test pgvector is working
            result = await conn.execute(text("SELECT '[1,2,3]'::vector"))
            print("✅ pgvector extension is working")
            
            # Check that tables exist
            table_count = await conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name LIKE '%documents%'
            """))
            count = table_count.scalar()
            
            if count > 0:
                print(f"✅ Database tables verified ({count} document-related tables found)")
            else:
                print("❌ No tables found")
                
            # Check legal theories
            theory_count = await conn.execute(text("SELECT COUNT(*) FROM legal_theories"))
            count = theory_count.scalar()
            print(f"✅ Legal theories seeded: {count} theories available")
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            raise
    
    await engine.dispose()


async def main():
    """Main initialization function"""
    print("🚀 Starting LeMCS Database Initialization")
    print("=" * 50)
    
    try:
        # Step 1: Create database if needed
        print("\n📁 Step 1: Database Creation")
        await create_database_if_not_exists()
        
        # Step 2: Set up extensions
        print("\n🔧 Step 2: Setting up PostgreSQL Extensions")
        await setup_pgvector_extension()
        
        # Step 3: Create tables
        print("\n📋 Step 3: Creating Database Tables")
        await create_tables()
        
        # Step 4: Create indexes
        print("\n⚡ Step 4: Creating Performance Indexes")
        await create_indexes()
        
        # Step 5: Seed data
        print("\n🌱 Step 5: Seeding Reference Data")
        await seed_legal_theories()
        
        # Step 6: Verify setup
        print("\n✅ Step 6: Verifying Setup")
        await verify_setup()
        
        print("\n" + "=" * 50)
        print("🎉 LeMCS Database Setup Complete!")
        print("📊 Your database is ready for legal document consolidation")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())