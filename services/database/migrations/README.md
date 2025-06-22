# Database Migrations

This directory contains SQL migration files for schema changes.

## Naming Convention
Migrations should be named with a timestamp prefix:
- `YYYYMMDD_HHMMSS_description.sql`
- Example: `20241222_143000_add_user_settings_table.sql`

## Migration Process
1. Create new migration file in this directory
2. Run `npm run migrate:schema` to apply migrations and update Zero permissions
3. For production: `npm run migrate:schema:prod`

## Notes
- Migrations are run in alphabetical order
- Always test migrations on development first
- Zero permissions are automatically redeployed after migrations
- The migrate-schema.sh script handles both database changes and Zero updates