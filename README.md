# Overheating Analysis Dashboard

Interactive visualization dashboard for analyzing building overheating hours across different time periods (Summer, July, Peak Week, Peak Day).

## Features

- **Interactive Filters**: Filter simulations by CDD10, Design Temperature (Tdb 2.5%), and FDWR values
- **Complementary CDF Plots**: Shows proportion of simulations exceeding different overheating thresholds
- **Multiple Time Scales**: Compare overheating patterns across:
  - Summer (May-September)
  - July only
  - Peak Week (highest 7-day period)
  - Peak Day (highest 24-hour period)
- **Real-time Statistics**: Mean, median, and 90th percentile values update with filters

## Data Files

The deployment folder contains:
- `app.py` - Main Dash application
- `parametric_results.csv` - Simulation results (168,007 simulations)
- `climate_zone_cities.json` - City to weather station mapping
- `cwec_climate_data.csv` - Climate data (CDD10, design temperatures)
- `requirements.txt` - Python dependencies
- `render.yaml` - Render.com configuration

## Local Testing

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://localhost:8050 in your browser.

## Deploy to Render.com

### Option 1: Using Render Dashboard (Recommended)

1. **Create Account**: Go to https://render.com and sign up (free tier available)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Choose "Deploy from folder" or connect to GitHub repository

3. **Configure Service**:
   - **Name**: `overheating-analysis` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server --bind 0.0.0.0:$PORT`
   - **Instance Type**: `Free` (or paid for better performance)

4. **Upload Files**:
   - If deploying from folder, upload all files in this deployment directory
   - If using Git, push this deployment folder to your repository

5. **Deploy**: Click "Create Web Service" and wait for deployment (3-5 minutes)

6. **Access**: Render will provide a URL like `https://overheating-analysis.onrender.com`

### Option 2: Using GitHub + Render

1. **Create GitHub Repository**:
   ```bash
   cd deployment
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/overheating-dashboard.git
   git push -u origin main
   ```

2. **Connect to Render**:
   - In Render dashboard, click "New +" → "Web Service"
   - Select "Connect a repository"
   - Authorize GitHub and select your repository
   - Render will auto-detect the `render.yaml` configuration

3. **Deploy**: Click "Create Web Service"

### Option 3: Using Render Blueprint (render.yaml)

The included `render.yaml` file contains all configuration. Simply:

1. Push code to GitHub
2. In Render dashboard, click "New +" → "Blueprint"
3. Select your repository
4. Render will read `render.yaml` and set up automatically

## Performance Notes

- **Free Tier**: Adequate for 10-20 concurrent users. Service spins down after 15 minutes of inactivity (30-60 second cold start)
- **Starter Tier ($7/mo)**: Recommended for regular use, no cold starts, better performance
- **Data Size**: CSV file is ~50MB. First load takes 10-15 seconds to read data.

## Troubleshooting

### App Won't Start
- Check logs in Render dashboard
- Verify all data files are uploaded
- Ensure `gunicorn` is in requirements.txt

### Slow Performance
- Free tier has limited resources
- Consider upgrading to Starter tier
- Data loads on startup, subsequent page interactions are fast

### Data Not Loading
- Verify CSV files are in the same directory as app.py
- Check file paths in app.py (should use `script_dir / 'filename'`)

## Sharing with Users

Once deployed, simply share the Render URL:
- **Example**: `https://overheating-analysis.onrender.com`
- Users can access from any device with a web browser
- No installation required
- Fully interactive (sliders, dropdowns all work)

## Updating the Dashboard

To update data or code:

1. **If using Git**:
   ```bash
   git add .
   git commit -m "Update data/code"
   git push
   ```
   Render will auto-deploy changes

2. **If using dashboard upload**:
   - Upload new files in Render dashboard
   - Click "Manual Deploy" → "Deploy latest commit"

## Cost Estimate

- **Free Tier**: $0/month (includes 750 hours/month)
- **Starter**: $7/month (no cold starts, better performance)
- **Standard**: $25/month (for high traffic)

For occasional use by 5-10 users, **Free tier is sufficient**.

## Support

For issues with:
- **Render deployment**: https://render.com/docs
- **Dashboard code**: Check app.py comments or modify as needed
- **Data questions**: Review CSV column definitions in source data

## License

This dashboard is for internal use. Data files contain simulation results proprietary to the research project.
