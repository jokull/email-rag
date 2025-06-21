import * as dotenv from 'dotenv';
import { createZeroServer } from '@rocicorp/zero';
import { schema } from './schema';

dotenv.config();

interface Config {
  port: number;
  databaseUrl: string;
  zeroAuth?: {
    secret: string;
  };
}

function getConfig(): Config {
  const requiredEnvVars = ['DATABASE_URL'];
  
  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      throw new Error(`Missing required environment variable: ${envVar}`);
    }
  }

  return {
    port: parseInt(process.env.PORT || '3000'),
    databaseUrl: process.env.DATABASE_URL!,
    zeroAuth: process.env.ZERO_AUTH_SECRET ? {
      secret: process.env.ZERO_AUTH_SECRET
    } : undefined,
  };
}

async function main() {
  const config = getConfig();
  
  console.log('Starting Zero server...');
  console.log(`Port: ${config.port}`);
  console.log(`Database: ${config.databaseUrl.replace(/:[^:@]*@/, ':***@')}`);

  const server = await createZeroServer({
    schema,
    upstream: {
      db: config.databaseUrl,
    },
    port: config.port,
    // Optional: Add authentication
    ...(config.zeroAuth ? {
      auth: {
        secret: config.zeroAuth.secret,
      }
    } : {}),
  });

  console.log(`Zero server running on port ${config.port}`);

  // Handle graceful shutdown
  process.on('SIGTERM', async () => {
    console.log('Received SIGTERM, shutting down gracefully...');
    await server.close();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    console.log('Received SIGINT, shutting down gracefully...');
    await server.close();
    process.exit(0);
  });
}

if (require.main === module) {
  main().catch((error) => {
    console.error('Failed to start Zero server:', error);
    process.exit(1);
  });
}