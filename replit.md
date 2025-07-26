# Lung Cancer Detection AI

## Overview

This is a Streamlit-based web application for lung cancer detection using AI. The application provides a user-friendly interface for uploading medical images (CT scans, X-rays) and analyzing them for potential signs of lung cancer. It includes mock AI models, image preprocessing capabilities, visualization tools, and sample data for demonstration purposes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Layout**: Wide layout with responsive columns
- **UI Components**: File upload, image display, interactive controls, charts and visualizations
- **Session Management**: Streamlit session state for analysis history tracking

### Backend Architecture
- **Core Framework**: Python-based modular design
- **Model Layer**: Mock AI models simulating cancer detection with configurable accuracy
- **Processing Pipeline**: Image preprocessing, enhancement, and analysis workflow
- **Data Layer**: In-memory session state for temporary data storage

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Entry point and UI orchestration
- **Features**: File upload interface, model selection, results display
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

### 5. Utility Functions (utils.py)
- **DICOM Processing**: Medical image file handling and metadata display
- **Analysis History**: Session-based tracking of previous analyses
- **Confidence Calculations**: Prediction confidence scoring

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
7. **History Tracking**: Analysis results stored in session state

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: AI model framework (for mock model structure)
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **OpenCV**: Computer vision operations
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation

### Medical Imaging
- **PyDICOM**: DICOM medical image format support
- **Scikit-image**: Advanced image processing

### Image Enhancement
- **PIL ImageEnhance**: Basic image enhancement
- **Scikit-image exposure**: Advanced exposure correction

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Dependencies**: All requirements installable via pip
- **No External Services**: Self-contained application without external API dependencies

### Production Considerations
- **Scaling**: Stateless design allows horizontal scaling
- **Security**: No persistent data storage, session-based state management
- **Medical Compliance**: Mock data only - requires real model integration for medical use

### Architecture Benefits
1. **Modularity**: Clear separation allows easy component updates
2. **Extensibility**: Mock model interface enables real AI model integration
3. **Educational Value**: Sample data system provides safe testing environment
4. **Medical Standards**: DICOM support for standard medical imaging workflows

### Technical Decisions

**Mock Model Approach**: Chosen to create a functional demo without requiring large AI model files or training data. Allows focus on UI/UX and application architecture.

**Streamlit Framework**: Selected for rapid prototyping and built-in web interface capabilities. Provides excellent balance of functionality and simplicity.

**Modular Design**: Enables independent development and testing of components. Facilitates future integration of real AI models.

**Session State Management**: Uses Streamlit's built-in session state for temporary data storage, avoiding need for external database in demo environment.