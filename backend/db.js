import { MongoClient } from "mongodb";

const DEFAULT_DB_NAME = process.env.MONGODB_DB_NAME || "NumberPredictor";
const HISTORY_COLLECTION =
  process.env.MONGODB_COLLECTION || "predictionHistory";
const MONGODB_URI = process.env.MONGODB_URI || "";
const MAX_HISTORY = Number(process.env.MAX_HISTORY || 18);

let client;
let collection;
let databaseState = {
  enabled: Boolean(MONGODB_URI),
  connected: false,
  mode: MONGODB_URI ? "disconnected" : "memory-only",
  databaseName: DEFAULT_DB_NAME,
  collectionName: HISTORY_COLLECTION,
  message: MONGODB_URI
    ? "MongoDB not connected yet"
    : "MONGODB_URI not set; using in-memory history",
};

export function getDatabaseState() {
  return databaseState;
}

export async function connectToDatabase() {
  if (!MONGODB_URI) {
    return null;
  }

  if (collection) {
    return collection;
  }

  try {
    client = new MongoClient(MONGODB_URI, {
      appName: "NumberPredictor",
    });

    await client.connect();

    const db = client.db(DEFAULT_DB_NAME);
    collection = db.collection(HISTORY_COLLECTION);

    await collection.createIndex({ createdAt: -1 });
    await collection.createIndex({ id: 1 }, { unique: true });

    databaseState = {
      enabled: true,
      connected: true,
      mode: "mongodb-atlas",
      databaseName: DEFAULT_DB_NAME,
      collectionName: HISTORY_COLLECTION,
      message: `Connected to MongoDB database ${DEFAULT_DB_NAME}`,
    };

    return collection;
  } catch (error) {
    collection = undefined;
    if (client) {
      await client.close().catch(() => undefined);
    }
    client = undefined;
    databaseState = {
      enabled: true,
      connected: false,
      mode: "disconnected",
      databaseName: DEFAULT_DB_NAME,
      collectionName: HISTORY_COLLECTION,
      message: `MongoDB connection failed: ${error.message}`,
    };
    throw error;
  }
}

export async function disconnectFromDatabase() {
  if (client) {
    await client.close();
  }

  client = undefined;
  collection = undefined;
  databaseState = {
    ...databaseState,
    connected: false,
    mode: MONGODB_URI ? "disconnected" : "memory-only",
  };
}

export async function loadHistoryFromDatabase() {
  const activeCollection = await connectToDatabase();
  if (!activeCollection) {
    return [];
  }

  const history = await activeCollection
    .find({}, { projection: { _id: 0 } })
    .sort({ createdAt: -1 })
    .limit(MAX_HISTORY)
    .toArray();

  return history;
}

export async function persistHistoryEntry(entry) {
  const activeCollection = await connectToDatabase();
  if (!activeCollection) {
    return false;
  }

  await activeCollection.updateOne(
    { id: entry.id },
    { $set: entry },
    { upsert: true },
  );

  const staleEntries = await activeCollection
    .find({}, { projection: { _id: 1 } })
    .sort({ createdAt: -1 })
    .skip(MAX_HISTORY)
    .toArray();

  if (staleEntries.length > 0) {
    await activeCollection.deleteMany({
      _id: { $in: staleEntries.map((item) => item._id) },
    });
  }

  return true;
}
