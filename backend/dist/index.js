import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import foodLogsRouter from './routes/foodLogs.js';
import recipesRouter from './routes/recipes.js';
const app = express();
const PORT = process.env.PORT || 3000;
// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
// Routes
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'foodtracker-api' });
});
app.use('/api/food-logs', foodLogsRouter);
app.use('/api/recipes', recipesRouter);
// Error handling
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(err.status || 500).json({
        error: err.message || 'Internal server error',
    });
});
// Start server
app.listen(PORT, () => {
    console.log(`🚀 Food Tracker API running on http://localhost:${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/health`);
});
