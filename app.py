from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
import uuid
import os
from datetime import datetime
from decimal import Decimal
import traceback
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
BUCKET_NAME = os.environ.get('S3_BUCKET', "product-review-upload")
REVIEWS_TABLE = os.environ.get('DYNAMODB_TABLE', "ProductReviews")
USERS_TABLE = os.environ.get('USERS_TABLE', "ProductUsers")

# AWS clients
s3 = boto3.client('s3', region_name=AWS_REGION)
rekognition = boto3.client('rekognition', region_name=AWS_REGION)
comprehend = boto3.client('comprehend', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
reviews_table = dynamodb.Table(REVIEWS_TABLE)
users_table = dynamodb.Table(USERS_TABLE)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Defect keywords to identify issues
DEFECT_KEYWORDS = [
    'broken', 'damaged', 'crack', 'scratch', 'dent', 'stain', 
    'tear', 'rip', 'hole', 'defect', 'worn', 'faded', 'bent',
    'discolored', 'peeling', 'rusty', 'chipped'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    """Decorator to protect routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to protect admin routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or not session.get('is_admin', False):
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def analyze_sentiment(text):
    """Analyze sentiment using AWS Comprehend"""
    try:
        if not text or len(text.strip()) < 3:
            return 'NEUTRAL', 0.5
        
        response = comprehend.detect_sentiment(
            Text=text[:5000],  # Comprehend limit
            LanguageCode='en'
        )
        sentiment = response['Sentiment']
        scores = response['SentimentScore']
        
        max_score = max(scores[sentiment])
        return sentiment, max_score
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return 'NEUTRAL', 0.5

def detect_defects(labels):
    """Detect defects from Rekognition labels"""
    defects = []
    for label in labels:
        label_name = label['Name'].lower()
        if any(keyword in label_name for keyword in DEFECT_KEYWORDS):
            defects.append({
                'name': label['Name'],
                'confidence': label['Confidence']
            })
    return defects

def categorize_product(labels):
    """Categorize product based on labels"""
    categories = {
        'Electronics': ['phone', 'mobile', 'laptop', 'computer', 'tablet', 'electronics', 'device'],
        'Clothing': ['shirt', 'pants', 'dress', 'clothing', 'apparel', 'fabric', 'textile'],
        'Footwear': ['shoe', 'boot', 'sandal', 'sneaker', 'footwear'],
        'Accessories': ['bag', 'wallet', 'watch', 'jewelry', 'belt', 'sunglasses'],
        'Home & Kitchen': ['furniture', 'appliance', 'utensil', 'cookware', 'decor'],
        'Beauty': ['cosmetic', 'makeup', 'perfume', 'skincare'],
        'Sports': ['ball', 'equipment', 'fitness', 'sports', 'athletic']
    }
    
    for label in labels:
        label_name = label['Name'].lower()
        for category, keywords in categories.items():
            if any(keyword in label_name for keyword in keywords):
                return category
    
    return 'Other'

def init_aws_resources():
    """Initialize all AWS resources"""
    print("\n" + "="*60)
    print("Initializing AWS Resources...")
    print("="*60)
    
    try:
        # Create S3 bucket
        try:
            s3.head_bucket(Bucket=BUCKET_NAME)
            print(f"✓ S3 bucket '{BUCKET_NAME}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if AWS_REGION == 'ap-south-1':
                        s3.create_bucket(Bucket=BUCKET_NAME)
                    else:
                        s3.create_bucket(
                            Bucket=BUCKET_NAME,
                            CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                        )
                    print(f"✓ Created S3 bucket '{BUCKET_NAME}'")
                except ClientError as create_error:
                    print(f"✗ Failed to create S3 bucket: {str(create_error)}")
        
        # Check DynamoDB tables
        try:
            reviews_table.load()
            print(f"✓ DynamoDB table '{REVIEWS_TABLE}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"✗ DynamoDB table '{REVIEWS_TABLE}' does NOT exist")
                print(f"   Create with: Partition key: image_id (String)")
        
        try:
            users_table.load()
            print(f"✓ DynamoDB table '{USERS_TABLE}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"✗ DynamoDB table '{USERS_TABLE}' does NOT exist")
                print(f"   Create with: Partition key: username (String)")
        
        print("="*60)
        print("AWS Resource Initialization Complete")
        print("="*60 + "\n")
            
    except Exception as e:
        print(f"\n✗ AWS initialization error: {str(e)}")
        print(traceback.format_exc())
        print("="*60 + "\n")

# ==================== HTML PAGE ROUTES ====================

@app.route('/')
def home():
    """Home page - Landing page"""
    return render_template('home.html')

@app.route('/login')
def login_page():
    """Login page"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Register page"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with analytics"""
    try:
        username = session['username']
        is_admin = session.get('is_admin', False)
        
        # Get user's reviews or all reviews for admin
        if is_admin:
            response = reviews_table.scan(Limit=100)
            reviews = response.get('Items', [])
        else:
            response = reviews_table.scan(
                FilterExpression='uploaded_by = :user',
                ExpressionAttributeValues={':user': username},
                Limit=50
            )
            reviews = response.get('Items', [])
        
        # Calculate analytics
        total_reviews = len(reviews)
        defect_count = sum(1 for r in reviews if r.get('has_defects', False))
        
        # Sentiment distribution
        sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'MIXED': 0}
        for review in reviews:
            sentiment = review.get('sentiment', 'NEUTRAL')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Category distribution
        category_counts = {}
        for review in reviews:
            category = review.get('product_category', 'Other')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Recent reviews
        sorted_reviews = sorted(reviews, 
                               key=lambda x: x.get('uploaded_at', ''), 
                               reverse=True)[:10]
        
        analytics = {
            'username': username,
            'is_admin': is_admin,
            'total_reviews': total_reviews,
            'defect_count': defect_count,
            'defect_rate': round((defect_count / total_reviews * 100) if total_reviews > 0 else 0, 2),
            'sentiment_counts': sentiment_counts,
            'category_counts': category_counts,
            'recent_reviews': sorted_reviews
        }
        
        return render_template('dashboard.html', data=analytics)
        
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/upload')
@login_required
def upload_page():
    """Upload new product review page"""
    return render_template('upload.html', username=session['username'])

@app.route('/reviews')
@login_required
def reviews_list():
    """List all reviews"""
    try:
        username = session['username']
        is_admin = session.get('is_admin', False)
        
        if is_admin:
            response = reviews_table.scan()
        else:
            response = reviews_table.scan(
                FilterExpression='uploaded_by = :user',
                ExpressionAttributeValues={':user': username}
            )
        
        reviews = response.get('Items', [])
        sorted_reviews = sorted(reviews, 
                               key=lambda x: x.get('uploaded_at', ''), 
                               reverse=True)
        
        return render_template('reviews.html', 
                             reviews=sorted_reviews, 
                             is_admin=is_admin)
        
    except Exception as e:
        flash(f'Error loading reviews: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/review/<review_id>')
@login_required
def review_detail(review_id):
    """Detailed view of a single review"""
    try:
        response = reviews_table.get_item(Key={'image_id': review_id})
        
        if 'Item' not in response:
            flash('Review not found', 'error')
            return redirect(url_for('reviews_list'))
        
        review = response['Item']
        
        # Check access permissions
        if not session.get('is_admin', False) and review.get('uploaded_by') != session['username']:
            flash('Access denied', 'error')
            return redirect(url_for('reviews_list'))
        
        return render_template('review_detail.html', review=review)
        
    except Exception as e:
        flash(f'Error loading review: {str(e)}', 'error')
        return redirect(url_for('reviews_list'))

@app.route('/analytics')
@login_required
def analytics_page():
    """Advanced analytics page"""
    try:
        is_admin = session.get('is_admin', False)
        username = session['username']
        
        if is_admin:
            response = reviews_table.scan()
        else:
            response = reviews_table.scan(
                FilterExpression='uploaded_by = :user',
                ExpressionAttributeValues={':user': username}
            )
        
        reviews = response.get('Items', [])
        
        # Detailed analytics
        analytics = {
            'total_reviews': len(reviews),
            'total_defects': sum(1 for r in reviews if r.get('has_defects', False)),
            'avg_confidence': sum(r.get('avg_confidence', 0) for r in reviews) / len(reviews) if reviews else 0,
            'sentiment_breakdown': {},
            'category_breakdown': {},
            'defect_types': {},
            'timeline_data': []
        }
        
        # Sentiment breakdown
        for review in reviews:
            sentiment = review.get('sentiment', 'NEUTRAL')
            analytics['sentiment_breakdown'][sentiment] = analytics['sentiment_breakdown'].get(sentiment, 0) + 1
        
        # Category breakdown
        for review in reviews:
            category = review.get('product_category', 'Other')
            analytics['category_breakdown'][category] = analytics['category_breakdown'].get(category, 0) + 1
        
        # Defect types
        for review in reviews:
            if review.get('has_defects'):
                for defect in review.get('defects', []):
                    defect_name = defect.get('name', 'Unknown')
                    analytics['defect_types'][defect_name] = analytics['defect_types'].get(defect_name, 0) + 1
        
        return render_template('analytics.html', analytics=analytics, is_admin=is_admin)
        
    except Exception as e:
        flash(f'Error loading analytics: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    try:
        username = session['username']
        response = users_table.get_item(Key={'username': username})
        user = response.get('Item', {})
        
        # Get user's review count
        reviews_response = reviews_table.scan(
            FilterExpression='uploaded_by = :user',
            ExpressionAttributeValues={':user': username},
            Select='COUNT'
        )
        review_count = reviews_response.get('Count', 0)
        
        user['review_count'] = review_count
        
        return render_template('profile.html', user=user)
        
    except Exception as e:
        flash(f'Error loading profile: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

# ==================== API ROUTES ====================

@app.route('/api/register', methods=['POST'])
def api_register():
    """Register new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip().lower()
        password = data.get('password', '')
        email = data.get('email', '').strip().lower()
        
        if not username or not password or not email:
            return jsonify({'error': 'All fields required'}), 400
        
        # Check if user exists
        response = users_table.get_item(Key={'username': username})
        if 'Item' in response:
            return jsonify({'error': 'Username already exists'}), 400
        
        # Create user (in production, hash the password!)
        user_id = str(uuid.uuid4())
        users_table.put_item(
            Item={
                'username': username,
                'user_id': user_id,
                'email': email,
                'password': password,  # HASH THIS IN PRODUCTION!
                'is_admin': False,
                'created_at': datetime.now().isoformat(),
                'review_count': 0
            }
        )
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'redirect': url_for('login_page')
        }), 201
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip().lower()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Get user
        response = users_table.get_item(Key={'username': username})
        
        if 'Item' not in response:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = response['Item']
        
        # Check password (in production, use proper password hashing!)
        if user.get('password') != password:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Set session
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        session['is_admin'] = user.get('is_admin', False)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'username': user['username'],
            'is_admin': user.get('is_admin', False),
            'redirect': url_for('dashboard')
        }), 200
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/upload-review', methods=['POST'])
@login_required
def api_upload_review():
    """Upload product image and review"""
    try:
        print("\n[UPLOAD] Starting upload process...")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        review_text = request.form.get('review', '').strip()
        product_name = request.form.get('product_name', '').strip()
        customer_name = request.form.get('customer_name', session['username'])
        
        if not product_name:
            return jsonify({'error': 'Product name required'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Generate unique IDs
        image_id = str(uuid.uuid4())
        filename = f"products/{image_id}/{secure_filename(file.filename)}"
        
        # Upload to S3
        image_bytes = file.read()
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType=file.content_type
        )
        
        s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        print(f"[UPLOAD] Image uploaded to S3: {s3_url}")
        
        # Analyze with Rekognition
        print("[UPLOAD] Analyzing image with Rekognition...")
        rekognition_response = rekognition.detect_labels(
            Image={'S3Object': {'Bucket': BUCKET_NAME, 'Name': filename}},
            MaxLabels=15,
            MinConfidence=70
        )
        
        labels = rekognition_response.get('Labels', [])
        print(f"[UPLOAD] Detected {len(labels)} labels")
        
        # Process labels
        label_names = [label['Name'] for label in labels]
        confidences = [float(label['Confidence']) for label in labels]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Detect defects
        defects = detect_defects(labels)
        has_defects = len(defects) > 0
        
        # Categorize product
        product_category = categorize_product(labels)
        
        # Analyze sentiment
        sentiment = 'NEUTRAL'
        sentiment_score = 0.5
        if review_text:
            sentiment, sentiment_score = analyze_sentiment(review_text)
        
        print(f"[UPLOAD] Sentiment: {sentiment}, Category: {product_category}, Defects: {has_defects}")
        
        # Store in DynamoDB
        review_item = {
            'image_id': image_id,
            'customer_name': customer_name,
            'product_name': product_name,
            's3_url': s3_url,
            'labels': label_names,
            'confidence_scores': [Decimal(str(c)) for c in confidences],
            'avg_confidence': Decimal(str(round(avg_confidence, 2))),
            'defects': defects,
            'has_defects': has_defects,
            'product_category': product_category,
            'review_text': review_text,
            'sentiment': sentiment,
            'sentiment_score': Decimal(str(round(sentiment_score, 4))),
            'uploaded_by': session['username'],
            'uploaded_at': datetime.now().isoformat()
        }
        
        reviews_table.put_item(Item=review_item)
        
        # Update user review count
        users_table.update_item(
            Key={'username': session['username']},
            UpdateExpression='SET review_count = if_not_exists(review_count, :zero) + :inc',
            ExpressionAttributeValues={':zero': 0, ':inc': 1}
        )
        
        print(f"[UPLOAD] Review saved successfully: {image_id}")
        
        return jsonify({
            'success': True,
            'message': 'Review uploaded and analyzed successfully',
            'data': {
                'image_id': image_id,
                'product_category': product_category,
                'has_defects': has_defects,
                'defect_count': len(defects),
                'sentiment': sentiment,
                'labels': label_names[:5]
            },
            'redirect': url_for('review_detail', review_id=image_id)
        }), 201
        
    except ClientError as e:
        error_message = e.response['Error']['Message']
        print(f"[UPLOAD] AWS Error: {error_message}")
        return jsonify({'error': f'AWS Error: {error_message}'}), 500
    except Exception as e:
        print(f"[UPLOAD] Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/delete-review/<review_id>', methods=['DELETE'])
@login_required
def api_delete_review(review_id):
    """Delete a review"""
    try:
        # Get review
        response = reviews_table.get_item(Key={'image_id': review_id})
        
        if 'Item' not in response:
            return jsonify({'error': 'Review not found'}), 404
        
        review = response['Item']
        
        # Check permissions
        if not session.get('is_admin', False) and review.get('uploaded_by') != session['username']:
            return jsonify({'error': 'Access denied'}), 403
        
        # Delete from DynamoDB
        reviews_table.delete_item(Key={'image_id': review_id})
        
        # Optionally delete from S3 (commented out for safety)
        # s3_key = review['s3_url'].split(f'{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/')[1]
        # s3.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        
        return jsonify({
            'success': True,
            'message': 'Review deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"Delete error: {str(e)}")
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST', 'GET'])
def api_logout():
    """Logout user"""
    session.clear()
    
    if request.method == 'GET':
        flash('You have been logged out', 'success')
        return redirect(url_for('home'))
    
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize AWS resources on startup
    init_aws_resources()
    
    print("\nStarting Flask application...")
    print(f"Server will be available at: http://0.0.0.0:5000")
    print("Press CTRL+C to quit\n")
    

    app.run(host='0.0.0.0', port=5000, debug=True)
