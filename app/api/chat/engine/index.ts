import {
  ContextChatEngine,
  LLM,
  MongoDBAtlasVectorSearch,
  serviceContextFromDefaults,
  VectorStoreIndex,
} from "llamaindex";
import { MongoClient } from "mongodb";
import { checkRequiredEnvVars, CHUNK_OVERLAP, CHUNK_SIZE } from "./shared.mjs";

async function getDataSource(llm: LLM) {
  checkRequiredEnvVars();
  const client = new MongoClient(process.env.MONGO_URI!);
  const serviceContext = serviceContextFromDefaults({
    llm,
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  const dbName = process.env.MONGODB_DATABASE;
  const collectionName = process.env.MONGODB_VECTORS;
  const indexName = process.env.MONGODB_VECTOR_INDEX;

  if (!dbName || !collectionName || !indexName) {
    throw new Error('One or more required environment variables are not set.');
  }
  const store = new MongoDBAtlasVectorSearch({
    mongodbClient: client,
    dbName: dbName,
    collectionName: collectionName,
    indexName: indexName,
  });

  return await VectorStoreIndex.fromVectorStore(store, serviceContext);
}

export async function createChatEngine(llm: LLM) {
  const index = await getDataSource(llm);
  const retriever = index.asRetriever({ similarityTopK: 3 });
  return new ContextChatEngine({
    chatModel: llm,
    retriever,
  });
}
