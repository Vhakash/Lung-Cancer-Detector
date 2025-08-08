# Lung Cancer Detection AI

## Overview

This is a Streamlit-based web application for lung cancer detection using AI. The application provides a user-friendly interface for uploading medical images (CT scans, X-rays) and analyzing them for potential signs of lung cancer. It includes mock AI models, image preprocessing capabilities, visualization tools, sample data for demonstration purposes, and **PostgreSQL database integration for persistent analysis history**.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Layout**: Wide layout with responsive columns
- **UI Components**: File upload, image display, interactive controls, charts and visualizations
- **Session Management**: Streamlit session state for temporary data + PostgreSQL for persistent history

### Backend Architecture
- **Core Framework**: Python-based modular design
- **Model Layer**: Mock AI models simulating cancer detection with configurable accuracy
- **Processing Pipeline**: Image preprocessing, enhancement, and analysis workflow
- **Data Layer**: PostgreSQL database for persistent analysis history + session state for temporary data
- **File Storage**: Local image storage in `history_images/` directory

### Database Architecture
- **Database**: PostgreSQL with psycopg2 connector
- **Schema**: Analysis history table with timestamps, predictions, model types, and image paths
- **Persistence**: Cross-session data retention and historical analysis tracking
- **Configuration**: Streamlit secrets management for database credentials

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Entry point and UI orchestration
- **Features**: File upload interface, model selection, results display, persistent history view
- **Architecture**: Component-based layout with column organization

### 2. AI Model System (model.py)
- **Mock Models**: Simulation of different model types (basic CNN, transfer learning)
- **Model Interface**: Standardized prediction interface
- **Performance Simulation**: Realistic accuracy and timing simulation

### 3. Image Processing Pipeline
- **Preprocessing (preprocessing.py)**: Image normalization, resizing, color channel management
- **Enhancement (image_enhancement.py)**: Multiple enhancement techniques (CLAHE, contrast, sharpening, etc.)
- **DICOM Support**: Medical image format handling and metadata extraction

### 4. Visualization System (visualization.py)
- **Prediction Visualization**: Confidence gauges, bar charts
- **Performance Metrics**: Model accuracy, precision, recall displays
- **Activation Maps**: Feature visualization for model interpretability

### 5. Database & Utility Functions (utils.py)
- **PostgreSQL Integration**: Database connection, schema initialization, and CRUD operations
- **Analysis History**: Persistent tracking of analyses across sessions
- **DICOM Processing**: Medical image file handling and metadata display
- **Confidence Calculations**: Prediction confidence scoring
- **Image Storage**: Local file system management for analyzed images

### 6. Medical Sample Data System (sample_data.py)
- **Medically Accurate Cases**: Six realistic medical cases with known diagnoses
- **Clinical Patterns**: Based on real radiological features from medical literature
- **Educational Cases**: Normal X-ray, Stage 1/3 lung cancer, normal CT, benign nodule, advanced cancer
- **Accurate Predictions**: Sample cases return clinically appropriate AI confidence levels

## Data Flow

1. **Image Input**: User uploads image or selects sample data
2. **Format Detection**: System identifies file type (DICOM, standard image formats)
3. **Preprocessing**: Image normalization, resizing, and color standardization
4. **Enhancement** (Optional): User-selected image enhancement techniques
5. **Model Prediction**: Mock AI model generates cancer detection probability
6. **Visualization**: Results displayed with confidence metrics and visualizations
7. **Database Storage**: Analysis results and images saved to PostgreSQL + local storage
8. **History Tracking**: Persistent analysis history accessible across sessions

## Database Schema

### History Table
```sql
CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    image_path VARCHAR(255),
    model_type VARCHAR(50),
    prediction_value REAL,
    prediction_label VARCHAR(50),
    confidence REAL,
    enhancement VARCHAR(50)
);
```

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: AI model framework (for mock model structure)
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **OpenCV**: Computer vision operations
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation

### Database
- **PostgreSQL**: Primary database system
- **psycopg2-binary**: PostgreSQL adapter for Python

### Medical Imaging
- **PyDICOM**: DICOM medical image format support
- **Scikit-image**: Advanced image processing

### Image Enhancement
- **PIL ImageEnhance**: Basic image enhancement
- **Scikit-image exposure**: Advanced exposure correction

## Setup and Configuration

### Security

**⚠️ SECURITY WARNING**: The `secrets.toml` file contains sensitive database credentials and should NEVER be committed to version control!

To properly configure your `secrets.toml` file:

1. **Copy the template file**:
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   ```
2. **Edit `.streamlit/secrets.toml` with your actual database credentials**:
   ```toml
   [postgres]
   host = "your_actual_host"
   port = 5432
   database = "your_actual_database"
   user = "your_actual_username"
   password = "your_actual_password"
   ```

### Database Setup
1. **PostgreSQL Installation**: Ensure PostgreSQL is installed and running
2. **Database Creation**: Create a database for the application
3. **Streamlit Secrets**: Configure database credentials in `.streamlit/secrets.toml` (see Security section above)

### Application Setup
1. **Install Dependencies**: `pip install -r requirements.txt` or use `pyproject.toml`
2. **Initialize Database**: Run the application once to auto-create the history table
3. **Create Directories**: The app will automatically create `history_images/` folder

## Deployment Strategy

### Development Environment
- **Platform**: Compatible with local development and cloud platforms
- **Dependencies**: All requirements installable via pip
- **Database**: Requires PostgreSQL instance (local or cloud)

### Production Considerations
- **Scaling**: Stateless application design with persistent database storage
- **Security**: Database credentials managed through Streamlit secrets
- **Data Persistence**: Analysis history and images stored permanently
- **Medical Compliance**: Mock data only - requires real model integration for medical use

### Architecture Benefits
1. **Modularity**: Clear separation allows easy component updates
2. **Extensibility**: Mock model interface enables real AI model integration
3. **Educational Value**: Sample data system provides safe testing environment
4. **Medical Standards**: DICOM support for standard medical imaging workflows
5. **Data Persistence**: PostgreSQL integration enables historical analysis and trends
6. **Scalability**: Database-backed architecture supports multiple users and large datasets

### Technical Decisions

**PostgreSQL Integration**: Added to provide persistent data storage, enabling analysis history tracking across sessions and supporting future analytics capabilities.

**Mock Model Approach**: Chosen to create a functional demo without requiring large AI model files or training data. Allows focus on UI/UX and application architecture.

**Streamlit Framework**: Selected for rapid prototyping and built-in web interface capabilities. Provides excellent balance of functionality and simplicity.

**Modular Design**: Enables independent development and testing of components. Facilitates future integration of real AI models.

**Hybrid Storage Strategy**: Combines PostgreSQL for structured data with local file storage for images, optimizing performance and storage costs.

## Features

### Core Functionality
- **Image Upload**: Support for DICOM and standard image formats
- **AI Analysis**: Mock models with realistic prediction confidence
- **Enhancement Tools**: Multiple image enhancement techniques
- **Visualization**: Comprehensive result visualization and model performance metrics

### Data Management
- **Persistent History**: All analyses stored in PostgreSQL database
- **Image Archive**: Analyzed images saved locally with database references
- **Cross-Session Access**: History available across application restarts
- **Data Export**: Analysis results can be queried and exported

### User Interface
- **Responsive Design**: Optimized for different screen sizes
- **Interactive Controls**: Real-time parameter adjustment
- **Medical Compliance**: Clear disclaimers and professional guidance
- **Educational Focus**: Sample cases with known diagnoses for learning