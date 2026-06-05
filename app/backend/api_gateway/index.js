require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { PrismaClient } = require('@prisma/client');

const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3000;

// ==========================================
// 1. CORS CONFIGURATION
// ==========================================
const allowedOrigins = [
  'http://localhost:3000', 
  'http://localhost:5173',
  'https://stgtdplatform.zxx.web.core.windows.net' // Placeholder for Azure Blob Storage
];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  credentials: true 
}));

app.use(express.json());

// ==========================================
// 2. MULTER SETUP
// ==========================================
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOADS_DIR);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});
const upload = multer({ storage: storage });

// ==========================================
// 3. API ROUTES
// ==========================================

// GET /api/history
app.use('/uploads', express.static(UPLOADS_DIR, {
  setHeaders: function (res, path, stat) {
    res.set('Access-Control-Allow-Origin', '*');
  }
}));

app.get('/api/history', async (req, res) => {
  try {
    const history = await prisma.predictionHistory.findMany({
      orderBy: { created_at: 'desc' },
      take: 50 // Limit to 50 latest records
    });
    res.json(history);
  } catch (error) {
    console.error("Error fetching history:", error);
    res.status(500).json({ error: "Failed to fetch prediction history" });
  }
});

// POST /api/predict
app.post('/api/predict', upload.single('file'), async (req, res) => {
  const startTime = Date.now();
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const pipelineType = req.body.pipeline_type || "morphological";
    const modelType = req.body.model_type || "rf";
    const absoluteImagePath = path.resolve(req.file.path);
    const absoluteOutputDir = path.resolve(UPLOADS_DIR);

    // Call Python ML Service
    let mlResponse;
    try {
      mlResponse = await axios.post('http://localhost:8000/predict', {
        image_path: absoluteImagePath,
        pipeline_type: pipelineType,
        output_dir: absoluteOutputDir,
        model_type: modelType
      });
    } catch (mlError) {
      console.error("ML Service Error:", mlError?.response?.data || mlError.message);
      return res.status(500).json({ 
        error: "Failed to process image via ML Service",
        details: mlError?.response?.data || mlError.message
      });
    }

    const { defect_type, confidence, features, images } = mlResponse.data;
    const inferenceTimeMs = Date.now() - startTime;

    // Relative URLs for Frontend / Nginx to serve
    // Because Nginx maps /uploads/ to the uploads directory
    const originalImageUrl = `/uploads/${req.file.filename}`;
    const overlayImageUrl = `/uploads/${images.overlay}`;

    // Save to PostgreSQL via Prisma
    const newRecord = await prisma.predictionHistory.create({
      data: {
        original_image_url: originalImageUrl,
        overlay_image_url: overlayImageUrl,
        pipeline_used: `${pipelineType} (${modelType})`,
        defect_type: defect_type,
        confidence_score: confidence,
        inference_time_ms: inferenceTimeMs
      }
    });

    res.json({
      success: true,
      data: {
        id: newRecord.id,
        defect_type,
        confidence,
        features,
        images: {
          original: originalImageUrl,
          overlay: overlayImageUrl
        },
        inference_time_ms: inferenceTimeMs
      }
    });

  } catch (error) {
    console.error("Unexpected error during /api/predict:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// ==========================================
// 4. START SERVER
// ==========================================
app.listen(PORT, () => {
  console.log(`API Gateway is running on http://localhost:${PORT}`);
});
