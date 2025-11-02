"""
Monte Carlo Demand Forecasting - Professional Edition
A sophisticated tool for demand forecasting using Monte Carlo simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Monte Carlo Forecasting Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Title styling */
    .stTitle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Headers with better colors */
    h1, h2, h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }
    
    /* Metric cards with shadow and hover effect */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.12);
    }
    
    /* Buttons with gradient and hover effects */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Info/Warning/Success boxes */
    div[data-baseweb="notification"] {
        border-radius: 12px;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 3px solid #667eea;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #667eea !important;
    }
    
    /* Radio buttons and checkboxes */
    div[data-testid="stRadio"] > div {
        background-color: white;
        padding: 0.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.05rem;
        color: #4a5568;
        border-radius: 8px 8px 0 0;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background-color: #f7fafc;
    }
    
    /* Number input and sliders */
    div[data-baseweb="input"] {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Download buttons special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Plotly charts container */
    div[data-testid="stPlotlyChart"] {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_data(n_periods=100):
    """Generate realistic sample demand data"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='D')
    
    # Generate demand with trend and seasonality
    trend = np.linspace(100, 150, n_periods)
    seasonality = 20 * np.sin(np.linspace(0, 4*np.pi, n_periods))
    noise = np.random.normal(0, 10, n_periods)
    demand = trend + seasonality + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    return pd.DataFrame({
        'Date': dates,
        'Demand': demand.astype(int)
    })

def analyze_data_quality(df):
    """Analyze the quality and characteristics of the data"""
    analysis = {
        'total_records': len(df),
        'missing_values': df['Demand'].isna().sum(),
        'negative_values': (df['Demand'] < 0).sum(),
        'zero_values': (df['Demand'] == 0).sum(),
        'mean': df['Demand'].mean(),
        'median': df['Demand'].median(),
        'std': df['Demand'].std(),
        'cv': df['Demand'].std() / df['Demand'].mean() if df['Demand'].mean() > 0 else 0,
        'min': df['Demand'].min(),
        'max': df['Demand'].max(),
        'skewness': df['Demand'].skew(),
        'kurtosis': df['Demand'].kurtosis()
    }
    return analysis

def fit_distribution(data):
    """Fit multiple distributions and select the best one"""
    distributions = {
        'Normal': stats.norm,
        'Lognormal': stats.lognorm,
        'Gamma': stats.gamma,
    }
    
    best_dist = None
    best_params = None
    best_ks_stat = np.inf
    best_name = None
    
    for name, dist in distributions.items():
        try:
            params = dist.fit(data)
            ks_stat, _ = stats.kstest(data, lambda x: dist.cdf(x, *params))
            
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_dist = dist
                best_params = params
                best_name = name
        except:
            continue
    
    return best_name, best_dist, best_params, best_ks_stat

def run_monte_carlo_simulation(df, n_simulations, confidence_level, time_horizon, distribution_type='normal'):
    """
    Advanced Monte Carlo simulation with multiple distribution options
    """
    demand_data = df['Demand'].values
    
    # Fit distribution
    if distribution_type == 'auto':
        dist_name, dist, params, ks_stat = fit_distribution(demand_data)
    else:
        mean_demand = demand_data.mean()
        std_demand = demand_data.std()
        dist_name = 'Normal'
        params = (mean_demand, std_demand)
    
    # Run simulations
    if dist_name == 'Normal':
        simulated_demands = np.random.normal(
            params[0], params[1], 
            size=(n_simulations, time_horizon)
        )
    elif dist_name == 'Lognormal':
        simulated_demands = np.random.lognormal(
            params[0], params[1], 
            size=(n_simulations, time_horizon)
        )
    else:
        simulated_demands = np.random.normal(
            demand_data.mean(), demand_data.std(), 
            size=(n_simulations, time_horizon)
        )
    
    # Ensure non-negative demands
    simulated_demands = np.maximum(simulated_demands, 0)
    
    # Calculate total demand for each simulation
    total_demands = simulated_demands.sum(axis=1)
    
    # Calculate statistics
    mean_forecast = total_demands.mean()
    std_forecast = total_demands.std()
    median_forecast = np.median(total_demands)
    
    # Calculate confidence intervals
    alpha = (100 - confidence_level) / 100
    lower_bound = np.percentile(total_demands, (alpha/2) * 100)
    upper_bound = np.percentile(total_demands, (1 - alpha/2) * 100)
    
    # Calculate percentiles for risk analysis
    percentiles = {
        'P10': np.percentile(total_demands, 10),
        'P25': np.percentile(total_demands, 25),
        'P50': np.percentile(total_demands, 50),
        'P75': np.percentile(total_demands, 75),
        'P90': np.percentile(total_demands, 90),
        'P95': np.percentile(total_demands, 95),
        'P99': np.percentile(total_demands, 99)
    }
    
    return {
        'simulated_demands': simulated_demands,
        'total_demands': total_demands,
        'mean_forecast': mean_forecast,
        'std_forecast': std_forecast,
        'median_forecast': median_forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'percentiles': percentiles,
        'distribution': dist_name,
        'confidence_level': confidence_level
    }

# ============================================================================
# MAIN APP
# ============================================================================

# Language Configuration
LANGUAGES = {
    'English': {
        'settings': '⚙️ Settings',
        'data_source': 'Data Source',
        'use_sample': 'Use Sample Data',
        'upload_data': 'Upload Historical Data',
        'upload_help': 'Upload a file with Date and Demand columns',
        'simulation_settings': 'Settings',
        'num_scenarios': 'Number of Scenarios to Test',
        'num_scenarios_help': 'How many \'what-if\' scenarios to run. More = more accurate (100,000 is recommended for best results)',
        'forecast_days': 'How Many Days to Forecast',
        'forecast_days_help': 'How far into the future do you want to predict? (Common: 7, 14, 30, or 60 days)',
        'confidence_level': 'Confidence Level (%)',
        'confidence_help': '95% means you can be 95% confident the actual demand will fall within the range shown',
        'analysis_method': 'Analysis Method',
        'standard': 'Standard (recommended)',
        'automatic': 'Automatic (advanced)',
        'method_help': 'Standard works for most cases',
        'about_tool': 'What This Tool Does',
        'about_desc': '''This tool analyzes your historical demand data and creates thousands of possible future scenarios to help you plan better.
        
**You get:**
- 📊 Expected demand forecast
- 🎯 Confidence ranges (best/worst case)
- ⚠️ Risk levels for planning
- 💾 Downloadable reports''',
        'main_title': '📈 Demand Forecasting Tool',
        'main_subtitle': 'Predict future demand with AI-powered confidence ranges',
        'language': 'Language / زبان',
        # Data Analysis
        'data_overview': 'Your Historical Data Overview',
        'days_of_data': 'Days of Data',
        'avg_daily_demand': 'Average Daily Demand',
        'variation': 'Variation',
        'stability': 'Stability',
        'data_loaded': 'Successfully loaded',
        'records': 'records',
        # Buttons
        'generate_forecast': '🚀 Generate Forecast',
        'start_over': '🔄 Start Over',
        'ready_new': 'Ready for new forecast!',
        'running_analysis': 'Analyzing your data and running',
        'scenarios': 'scenarios... This may take a few seconds...',
        'forecast_ready': 'Your forecast is ready! Scroll down to see results.',
        # Results
        'results_title': 'Results - What You Need To Know',
        'bottom_line': 'Bottom Line for the Next',
        'days': 'Days',
        'expected_daily': 'Expected Daily Demand',
        'around': 'Around',
        'units_per_day': 'units per day',
        'total_demand': 'Total Demand for Period',
        'units': 'units',
        'over': 'over',
        'how_confident': 'How Confident Are We?',
        'chance': 'chance',
        'daily_demand_between': 'daily demand will be between',
        'and': 'and',
        'total_demand_between': 'total demand will be between',
        'what_to_do': 'What Should You Do?',
        'conservative': 'Conservative approach',
        'plan_for': 'Plan for',
        'total_units': 'total units',
        'covers': 'covers',
        'of_scenarios': 'of scenarios',
        'balanced': 'Balanced approach',
        'average_case': 'average case',
        'safe': 'Safe approach',
        # Welcome Screen
        'welcome_title': 'Welcome! Let\'s Get Started',
        'welcome_subtitle': 'Upload your demand data or use sample data from the sidebar to begin forecasting',
        'how_to_use': 'How to Use This Tool',
        'step1_title': 'Upload Your Data',
        'step2_title': 'Adjust Settings',
        'step3_title': 'Get Your Forecast',
        # Charts and Views
        'detailed_numbers': 'Detailed Numbers',
        'view_forecast_as': 'View forecast as:',
        'total_demand_sum': 'Total Demand (Sum)',
        'daily_average_demand': 'Daily Average Demand',
        'both': 'Both',
        'total_over': 'Total Demand over',
        'mean_total': 'Mean Total',
        'median_total': 'Median Total',
        'lower': 'Lower',
        'upper': 'Upper',
        'daily_avg_per_day': 'Daily Average Demand (per day over',
        'mean_daily': 'Mean Daily',
        'median_daily': 'Median Daily',
        'comparison_historical': 'Comparison with Historical Data:',
        'historical_daily_avg': 'Historical Daily Average',
        'forecasted_daily_avg': 'Forecasted Daily Average',
        'difference': 'Difference',
        # Planning Scenarios
        'planning_scenarios': 'Planning Scenarios - Pick Your Comfort Level',
        'scenario': 'Scenario',
        'confidence_level_col': 'Confidence Level',
        'total_units_needed': 'Total Units Needed',
        'per_day': 'Per Day',
        'minimum_expected': 'Minimum Expected',
        'low_side': 'Low Side',
        'most_likely': 'Most Likely (50/50)',
        'high_side': 'High Side',
        'safe_planning': 'Safe Planning',
        'very_safe': 'Very Safe',
        'emergency_buffer': 'Emergency Buffer',
        'only_10_lower': 'Only 10% chance demand is lower',
        'only_25_lower': '25% chance demand is lower',
        'fifty_fifty': '50% chance above, 50% below',
        'only_75_lower': '75% chance demand is lower',
        'only_90_lower': '90% chance demand is lower',
        'only_95_lower': '95% chance demand is lower',
        'only_99_lower': '99% chance demand is lower',
        # Download
        'download_simulation': 'Download Simulation Results (CSV)',
        'download_summary': 'Download Summary (CSV)',
        'download_risk': 'Download Risk Analysis (CSV)',
        # Charts tabs
        'distribution': 'Distribution',
        'confidence_intervals': 'Confidence Intervals',
        'simulation_paths': 'Simulation Paths',
        'forecast_distribution': 'Forecast Distribution',
        'show_distribution_as': 'Show distribution as:',
        # Ready to forecast
        'ready_forecast': 'Ready to Get Your Forecast?',
        'click_below': 'Click the button below to analyze your data and generate demand predictions.',
        # Welcome screen detailed content
        'step1_content': '''**Your file should have 2 columns:**
- **Column 1:** Dates
- **Column 2:** Demand numbers

**✅ Accepted formats:**
- Excel (.xlsx)
- CSV (.csv)  
- Text (.txt)

💡 *Or use "Sample Data" to try it out!*''',
        'step2_content': '''**In the sidebar, set:**
- **How many days** to forecast
- **Confidence level** (95% is standard)

**✅ Default settings work great!**

💡 *Not sure? Just leave defaults as-is.*''',
        'step3_content': '''**Click "Generate Forecast" and get:**

✅ Expected demand forecast  
✅ Best/worst case ranges  
✅ Planning scenarios  
✅ Easy-to-read charts

💡 *Download results as CSV when done!*''',
        # FAQ
        'common_questions': 'Common Questions',
        'faq_data_need': 'What kind of data do I need?',
        'faq_what_tells': 'What will this tool tell me?',
        'faq_accuracy': 'How accurate is this?',
        'faq_what_do': 'What do I do with the results?',
        # Expandable sections
        'detailed_statistics': 'Detailed Statistics',
        'data_validation': 'Data Validation & Interpretation',
        'historical_pattern': 'Historical Demand Pattern',
        'time_series': 'Time Series',
        'statistics': 'Statistics',
        'detailed_summary': 'Detailed Summary',
        'simulation_validation': 'Simulation Validation & Methodology',
        'which_view': 'Which view should I use?',
        'export_results': 'Export Results',
        'distribution': 'Distribution',
        'params_changed': '⚠️ Settings have changed. Click "Generate Forecast" to update results.',
        # FAQ Content
        'faq_data_need_content': """You need historical demand data - basically a record of past sales, orders, or usage.

**Example:**
```
Date          Demand
2024-01-01    150
2024-01-02    145
2024-01-03    160
...
```

**Minimum:** At least 30 days of data (more is better!)""",
        'faq_what_tells_content': """This tool answers questions like:
- 📊 "How much demand should I expect next month?"
- 🎯 "What's the likely range (best/worst case)?"
- 📦 "How much inventory should I stock?"
- ⚠️ "What if demand is higher than expected?"

**You get clear answers** with confidence levels (e.g., "90% chance demand will be between X and Y").""",
        'faq_accuracy_content': """The forecast is based on your historical patterns. 

**More accurate if:**
✅ You have more historical data (60+ days ideal)
✅ Your demand is relatively stable
✅ No major changes expected (new products, market shifts, etc.)

**The tool tells you** how confident you can be in the results!""",
        'faq_what_do_content': """Use the forecast for:
- 📦 **Inventory planning** - How much stock to order
- 👥 **Staffing decisions** - How many people needed
- 💰 **Budgeting** - Expected revenue/costs
- 📊 **Capacity planning** - Resources needed

**Pick a scenario** that matches your risk tolerance (conservative, balanced, or safe)."""
    },
    'فارسی': {
        'settings': '⚙️ تنظیمات',
        'data_source': 'منبع داده',
        'use_sample': 'استفاده از داده نمونه',
        'upload_data': 'بارگذاری داده‌های تاریخی',
        'upload_help': 'فایلی با ستون‌های تاریخ و تقاضا بارگذاری کنید',
        'simulation_settings': 'تنظیمات',
        'num_scenarios': 'تعداد سناریوهای آزمایش',
        'num_scenarios_help': 'چند سناریو اجرا شود. بیشتر = دقیق‌تر (100,000 پیشنهاد می‌شود)',
        'forecast_days': 'چند روز پیش‌بینی شود',
        'forecast_days_help': 'چقدر به آینده پیش‌بینی می‌کنید؟ (معمول: 7، 14، 30، یا 60 روز)',
        'confidence_level': 'سطح اطمینان (%)',
        'confidence_help': '95٪ به معنای اطمینان 95٪ است که تقاضای واقعی در این محدوده خواهد بود',
        'analysis_method': 'روش تحلیل',
        'standard': 'استاندارد (پیشنهادی)',
        'automatic': 'خودکار (پیشرفته)',
        'method_help': 'استاندارد برای اکثر موارد کار می‌کند',
        'about_tool': 'این ابزار چه کاری انجام می‌دهد',
        'about_desc': '''این ابزار داده‌های تاریخی تقاضای شما را تجزیه و تحلیل کرده و هزاران سناریوی ممکن برای برنامه‌ریزی بهتر ایجاد می‌کند.

**شما دریافت می‌کنید:**
- 📊 پیش‌بینی تقاضای مورد انتظار
- 🎯 محدوده‌های اطمینان (بهترین/بدترین حالت)
- ⚠️ سطوح ریسک برای برنامه‌ریزی
- 💾 گزارش‌های قابل دانلود''',
        'main_title': '📈 ابزار پیش‌بینی تقاضا',
        'main_subtitle': 'پیش‌بینی تقاضای آینده با محدوده‌های اطمینان هوش مصنوعی',
        'language': 'Language / زبان',
        # Data Analysis
        'data_overview': 'نمای کلی داده‌های تاریخی شما',
        'days_of_data': 'روزهای داده',
        'avg_daily_demand': 'میانگین تقاضای روزانه',
        'variation': 'تغییرات',
        'stability': 'پایداری',
        'data_loaded': 'با موفقیت بارگذاری شد',
        'records': 'رکورد',
        # Buttons
        'generate_forecast': '🚀 ایجاد پیش‌بینی',
        'start_over': '🔄 شروع مجدد',
        'ready_new': 'آماده برای پیش‌بینی جدید!',
        'running_analysis': 'در حال تجزیه و تحلیل داده‌های شما و اجرای',
        'scenarios': 'سناریو... ممکن است چند ثانیه طول بکشد...',
        'forecast_ready': 'پیش‌بینی شما آماده است! برای مشاهده نتایج به پایین بروید.',
        # Results
        'results_title': 'نتایج - آنچه باید بدانید',
        'bottom_line': 'نتیجه نهایی برای',
        'days': 'روز',
        'expected_daily': 'تقاضای روزانه مورد انتظار',
        'around': 'حدود',
        'units_per_day': 'واحد در روز',
        'total_demand': 'تقاضای کل برای دوره',
        'units': 'واحد',
        'over': 'در طول',
        'how_confident': 'چقدر مطمئنیم؟',
        'chance': 'احتمال',
        'daily_demand_between': 'تقاضای روزانه بین',
        'and': 'و',
        'total_demand_between': 'تقاضای کل بین',
        'what_to_do': 'چه کاری باید انجام دهید؟',
        'conservative': 'رویکرد محافظه‌کارانه',
        'plan_for': 'برنامه‌ریزی برای',
        'total_units': 'کل واحدها',
        'covers': 'پوشش می‌دهد',
        'of_scenarios': 'از سناریوها',
        'balanced': 'رویکرد متعادل',
        'average_case': 'حالت میانگین',
        'safe': 'رویکرد ایمن',
        # Welcome Screen
        'welcome_title': 'خوش آمدید! بیایید شروع کنیم',
        'welcome_subtitle': 'داده‌های تقاضای خود را بارگذاری کنید یا از داده نمونه در نوار کناری استفاده کنید',
        'how_to_use': 'نحوه استفاده از این ابزار',
        'step1_title': 'داده‌های خود را بارگذاری کنید',
        'step2_title': 'تنظیمات را تنظیم کنید',
        'step3_title': 'پیش‌بینی خود را دریافت کنید',
        # Charts and Views
        'detailed_numbers': 'اعداد تفصیلی',
        'view_forecast_as': 'نمایش پیش‌بینی به صورت:',
        'total_demand_sum': 'تقاضای کل (مجموع)',
        'daily_average_demand': 'میانگین تقاضای روزانه',
        'both': 'هر دو',
        'total_over': 'تقاضای کل در طول',
        'mean_total': 'میانگین کل',
        'median_total': 'میانه کل',
        'lower': 'پایین‌تر',
        'upper': 'بالاتر',
        'daily_avg_per_day': 'میانگین تقاضای روزانه (در هر روز در طول',
        'mean_daily': 'میانگین روزانه',
        'median_daily': 'میانه روزانه',
        'comparison_historical': 'مقایسه با داده‌های تاریخی:',
        'historical_daily_avg': 'میانگین روزانه تاریخی',
        'forecasted_daily_avg': 'میانگین روزانه پیش‌بینی شده',
        'difference': 'تفاوت',
        # Planning Scenarios
        'planning_scenarios': 'سناریوهای برنامه‌ریزی - سطح راحتی خود را انتخاب کنید',
        'scenario': 'سناریو',
        'confidence_level_col': 'سطح اطمینان',
        'total_units_needed': 'کل واحدهای مورد نیاز',
        'per_day': 'در روز',
        'minimum_expected': '🟢 حداقل مورد انتظار',
        'low_side': '🟡 طرف پایین',
        'most_likely': '🟢 محتمل‌ترین (50/50)',
        'high_side': '🟡 طرف بالا',
        'safe_planning': '🟠 برنامه‌ریزی ایمن',
        'very_safe': '🟠 بسیار ایمن',
        'emergency_buffer': '🔴 بافر اضطراری',
        'only_10_lower': 'فقط 10٪ احتمال که تقاضا کمتر باشد',
        'only_25_lower': '25٪ احتمال که تقاضا کمتر باشد',
        'fifty_fifty': '50٪ احتمال بالاتر، 50٪ پایین‌تر',
        'only_75_lower': '75٪ احتمال که تقاضا کمتر باشد',
        'only_90_lower': '90٪ احتمال که تقاضا کمتر باشد',
        'only_95_lower': '95٪ احتمال که تقاضا کمتر باشد',
        'only_99_lower': '99٪ احتمال که تقاضا کمتر باشد',
        # Download
        'download_simulation': '📥 دانلود نتایج شبیه‌سازی (CSV)',
        'download_summary': '📥 دانلود خلاصه (CSV)',
        'download_risk': '📥 دانلود تحلیل ریسک (CSV)',
        # Charts tabs
        'distribution': 'توزیع',
        'confidence_intervals': 'فواصل اطمینان',
        'simulation_paths': 'مسیرهای شبیه‌سازی',
        'forecast_distribution': 'توزیع پیش‌بینی',
        'show_distribution_as': 'نمایش توزیع به صورت:',
        # Ready to forecast
        'ready_forecast': 'آماده دریافت پیش‌بینی هستید؟',
        'click_below': 'روی دکمه زیر کلیک کنید تا داده‌های شما را تجزیه و تحلیل کرده و پیش‌بینی تقاضا را ایجاد کنید.',
        # Welcome screen detailed content
        'step1_content': '''**فایل شما باید 2 ستون داشته باشد:**
- **ستون 1:** تاریخ‌ها
- **ستون 2:** اعداد تقاضا

**✅ فرمت‌های پذیرفته شده:**
- Excel (.xlsx)
- CSV (.csv)  
- Text (.txt)

💡 *یا از "داده نمونه" برای آزمایش استفاده کنید!*''',
        'step2_content': '''**در نوار کناری، تنظیم کنید:**
- **چند روز** برای پیش‌بینی
- **سطح اطمینان** (95٪ استاندارد است)

**✅ تنظیمات پیش‌فرض عالی کار می‌کنند!**

💡 *مطمئن نیستید؟ پیش‌فرض‌ها را همان‌طور بگذارید.*''',
        'step3_content': '''**روی "ایجاد پیش‌بینی" کلیک کنید و دریافت کنید:**

✅ پیش‌بینی تقاضای مورد انتظار  
✅ محدوده‌های بهترین/بدترین حالت  
✅ سناریوهای برنامه‌ریزی  
✅ نمودارهای آسان برای خواندن

💡 *نتایج را به صورت CSV دانلود کنید!*''',
        # FAQ
        'common_questions': 'سوالات متداول',
        'faq_data_need': 'به چه نوع داده‌ای نیاز دارم؟',
        'faq_what_tells': 'این ابزار به من چه می‌گوید؟',
        'faq_accuracy': 'این چقدر دقیق است؟',
        'faq_what_do': 'با نتایج چه کار کنم؟',
        # Expandable sections
        'detailed_statistics': 'آمار تفصیلی',
        'data_validation': 'اعتبارسنجی و تفسیر داده‌ها',
        'historical_pattern': 'الگوی تقاضای تاریخی',
        'time_series': 'سری زمانی',
        'statistics': 'آمار',
        'detailed_summary': 'خلاصه تفصیلی',
        'simulation_validation': 'اعتبارسنجی و روش‌شناسی شبیه‌سازی',
        'which_view': 'از کدام نما استفاده کنم؟',
        'export_results': 'صادرات نتایج',
        'distribution': 'توزیع',
        'params_changed': '⚠️ تنظیمات تغییر کرده‌اند. برای به‌روزرسانی نتایج روی "ایجاد پیش‌بینی" کلیک کنید.',
        # FAQ Content
        'faq_data_need_content': """شما به داده‌های تاریخی تقاضا نیاز دارید - در واقع سابقه فروش، سفارشات یا مصرف گذشته.

**مثال:**
```
تاریخ          تقاضا
2024-01-01    150
2024-01-02    145
2024-01-03    160
...
```

**حداقل:** حداقل 30 روز داده (هرچه بیشتر بهتر!)""",
        'faq_what_tells_content': """این ابزار به سوالاتی مانند این پاسخ می‌دهد:
- 📊 "چقدر تقاضا باید برای ماه آینده انتظار داشته باشم؟"
- 🎯 "محدوده احتمالی (بهترین/بدترین حالت) چقدر است؟"
- 📦 "چقدر موجودی باید ذخیره کنم؟"
- ⚠️ "اگر تقاضا بیشتر از انتظار باشد چه؟"

**پاسخ‌های واضحی دریافت می‌کنید** با سطوح اطمینان (مثلاً "90٪ احتمال دارد تقاضا بین X و Y باشد").""",
        'faq_accuracy_content': """پیش‌بینی بر اساس الگوهای تاریخی شما است.

**دقیق‌تر اگر:**
✅ داده‌های تاریخی بیشتری داشته باشید (60+ روز ایده‌آل است)
✅ تقاضای شما نسبتاً پایدار باشد
✅ تغییرات عمده‌ای انتظار نرود (محصولات جدید، تغییرات بازار و غیره)

**ابزار به شما می‌گوید** تا چه حد می‌توانید به نتایج اطمینان داشته باشید!""",
        'faq_what_do_content': """از پیش‌بینی برای موارد زیر استفاده کنید:
- 📦 **برنامه‌ریزی موجودی** - چقدر موجودی سفارش دهید
- 👥 **تصمیمات نیروی انسانی** - چند نفر نیاز است
- 💰 **بودجه‌بندی** - درآمد/هزینه‌های مورد انتظار
- 📊 **برنامه‌ریزی ظرفیت** - منابع مورد نیاز

**یک سناریو انتخاب کنید** که با تحمل ریسک شما مطابقت داشته باشد (محافظه‌کارانه، متعادل یا ایمن)."""
    }
}

# Get language from session state if sidebar hasn't run yet
if 'language_selector' not in st.session_state:
    st.session_state['language_selector'] = 'English'

current_lang = LANGUAGES[st.session_state.get('language_selector', 'English')]

# Header with enhanced styling
st.markdown(f"""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        {current_lang['main_title']}
    </h1>
    <p style='font-size: 1.3rem; color: #718096; margin-top: 0;'>
        {current_lang['main_subtitle']}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")  # Spacer

# Sidebar Configuration
with st.sidebar:
    # Language selector at the top
    selected_language = st.selectbox(
        "Language / زبان",
        options=['English', 'فارسی'],
        index=0,
        key='language_selector'
    )
    
    lang = LANGUAGES[selected_language]
    
    # Apply RTL for Farsi
    if selected_language == 'فارسی':
        st.markdown("""
        <style>
        .main * {
            direction: rtl;
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem; border-bottom: 3px solid #667eea;'>
        <h2 style='color: #667eea !important; margin: 0; font-size: 2rem;'>{lang['settings']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(lang['data_source'])
    
    use_sample = st.checkbox(lang['use_sample'], value=False, 
                             help=lang['upload_help'])
    
    if not use_sample:
        uploaded_file = st.file_uploader(
            lang['upload_data'],
            type=['csv', 'xlsx', 'txt'],
            help=lang['upload_help']
        )
    else:
        uploaded_file = None
    
    st.markdown("---")
    st.subheader(lang['simulation_settings'])
    
    n_simulations = st.number_input(
        lang['num_scenarios'],
        min_value=1000,
        max_value=500000,
        value=100000,
        step=5000,
        help=lang['num_scenarios_help']
    )
    
    time_horizon = st.number_input(
        lang['forecast_days'],
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help=lang['forecast_days_help']
    )
    
    confidence_level = st.slider(
        lang['confidence_level'],
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help=lang['confidence_help']
    )
    
    distribution_type = st.selectbox(
        lang['analysis_method'],
        options=['normal', 'auto'],
        index=0,
        format_func=lambda x: lang['standard'] if x == 'normal' else lang['automatic'],
        help=lang['method_help']
    )
    
    st.markdown("---")
    st.markdown(f"**📖 {lang['about_tool']}**")
    st.markdown(lang['about_desc'])

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

df = None

# Load data
if use_sample:
    df = generate_sample_data()
    st.info("📊 Using sample data for demonstration")
elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate and clean data
        if df.shape[1] < 2:
            st.error("❌ File must contain at least two columns (Date and Demand)")
            df = None
        else:
            # Use first two columns
            df.columns = ['Date', 'Demand'] + list(df.columns[2:]) if df.shape[1] > 2 else ['Date', 'Demand']
            df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
            df = df.dropna(subset=['Demand'])
            
            if len(df) == 0:
                st.error("❌ No valid demand data found")
                df = None
            else:
                st.success(f"✅ Successfully loaded {len(df)} records")
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        df = None

# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

if df is not None:
    
    # Data Quality Analysis
    st.header(f"📊 {current_lang['data_overview']}")
    
    analysis = analyze_data_quality(df)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(current_lang['days_of_data'], f"{analysis['total_records']:,}", help="More data = more accurate forecast")
    with col2:
        st.metric(current_lang['avg_daily_demand'], f"{analysis['mean']:.1f}", help="Your typical daily demand")
    with col3:
        st.metric(current_lang['variation'], f"±{analysis['std']:.1f}", help="How much demand fluctuates daily")
    with col4:
        cv_pct = analysis['cv']*100
        cv_emoji = "🟢" if cv_pct < 10 else "🟡" if cv_pct < 20 else "🟠"
        st.metric(current_lang['stability'], f"{cv_emoji} {cv_pct:.1f}%", 
                 help="Lower = more predictable. <10% is very stable, <20% is good")
    
    # Detailed Statistics
    with st.expander(f"📈 {current_lang['detailed_statistics']}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            stats_df = pd.DataFrame({
                'Metric': ['Min', 'Max', 'Median', 'Skewness', 'Kurtosis'],
                'Value': [
                    f"{analysis['min']:.2f}",
                    f"{analysis['max']:.2f}",
                    f"{analysis['median']:.2f}",
                    f"{analysis['skewness']:.2f}",
                    f"{analysis['kurtosis']:.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            quality_df = pd.DataFrame({
                'Quality Check': ['Missing Values', 'Negative Values', 'Zero Values'],
                'Count': [
                    analysis['missing_values'],
                    analysis['negative_values'],
                    analysis['zero_values']
                ]
            })
            st.dataframe(quality_df, hide_index=True, use_container_width=True)
    
    # Validation & Interpretation
    with st.expander(f"✅ {current_lang['data_validation']}", expanded=False):
        st.markdown("#### What do these numbers mean?")
        
        # CV Interpretation
        cv_pct = analysis['cv'] * 100
        if cv_pct < 10:
            cv_status = "🟢 Excellent - Very stable demand"
        elif cv_pct < 20:
            cv_status = "🟡 Good - Moderate variability"
        elif cv_pct < 30:
            cv_status = "🟠 Fair - High variability"
        else:
            cv_status = "🔴 Poor - Very high variability"
        
        st.markdown(f"""
        **Coefficient of Variation (CV): {cv_pct:.1f}%** - {cv_status}
        - CV shows demand variability relative to mean
        - Your CV of {cv_pct:.1f}% means demand varies by ±{cv_pct:.1f}% on average
        """)
        
        # Skewness interpretation
        skew = analysis['skewness']
        if abs(skew) < 0.5:
            skew_status = "🟢 Symmetric distribution (Normal-like)"
        elif abs(skew) < 1:
            skew_status = "🟡 Moderately skewed"
        else:
            skew_status = "🔴 Highly skewed"
        
        st.markdown(f"""
        **Skewness: {skew:.2f}** - {skew_status}
        - Measures asymmetry of the distribution
        - Close to 0 = symmetric, > 0 = right tail, < 0 = left tail
        """)
        
        # Sample calculation verification
        st.markdown("#### 🧮 Verify Calculations (Manual Check)")
        st.code(f"""
Mean = Sum of all demands / Number of records
     = {df['Demand'].sum():.2f} / {len(df)}
     = {analysis['mean']:.2f} ✓

Standard Deviation = √(Σ(x - mean)² / (n-1))
                   = {analysis['std']:.2f} ✓

CV = (Std Dev / Mean) × 100
   = ({analysis['std']:.2f} / {analysis['mean']:.2f}) × 100
   = {cv_pct:.2f}% ✓
        """, language="text")
        
        # Show raw data sample
        st.markdown("#### 📊 Raw Data Sample (First 10 & Last 10 Records)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**First 10 Records**")
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Last 10 Records**")
            st.dataframe(df.tail(10), hide_index=True, use_container_width=True)
    
    # Historical Data Visualization
    st.subheader(f"📉 {current_lang['historical_pattern']}")
    
    tab1, tab2, tab3 = st.tabs([current_lang['time_series'], current_lang['distribution'], current_lang['statistics']])
    
    with tab1:
        # Time series plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['Demand'],
            mode='lines',
            name='Demand',
            line=dict(color='#3b82f6', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=[df['Demand'].mean()] * len(df),
            mode='lines',
            name='Mean',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Historical Demand Over Time",
            xaxis_title="Time Period",
            yaxis_title="Demand",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Demand'],
            nbinsx=50,
            name='Demand',
            marker_color='#3b82f6'
        ))
        fig.update_layout(
            title="Demand Distribution",
            xaxis_title="Demand",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Box plot and violin plot
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df['Demand'], name='Demand', marker_color='#3b82f6'))
            fig.update_layout(title="Box Plot", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Violin(y=df['Demand'], name='Demand', marker_color='#10b981'))
            fig.update_layout(title="Violin Plot", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # RUN SIMULATION
    # ============================================================================
    
    st.header(f"🎲 {current_lang['ready_forecast']}")
    st.markdown(current_lang['click_below'])
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_button = st.button(current_lang['generate_forecast'], type="primary", use_container_width=True)
    with col2:
        if st.button(current_lang['start_over'], use_container_width=True):
            if 'results' in st.session_state:
                del st.session_state['results']
                st.success(current_lang['ready_new'])
    with col3:
        pass
    
    if run_button:
        with st.spinner(f'🔄 {current_lang["running_analysis"]} {n_simulations:,} {current_lang["scenarios"]}'):
            results = run_monte_carlo_simulation(
                df, n_simulations, confidence_level, time_horizon, distribution_type
            )
            st.session_state['results'] = results
            st.session_state['params'] = {
                'n_simulations': n_simulations,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            st.success(f"✅ {current_lang['forecast_ready']}")
    
    # ============================================================================
    # DISPLAY RESULTS
    # ============================================================================
    
    if 'results' in st.session_state:
        st.header(f"📊 {current_lang['results_title']}")
        
        results = st.session_state['results']
        params = st.session_state['params']
        
        # Check if parameters have changed
        if (params['time_horizon'] != time_horizon or 
            params['confidence_level'] != confidence_level or 
            params['n_simulations'] != n_simulations):
            st.warning(current_lang['params_changed'])  
        
        # ==============================================================
        # QUICK SUMMARY - Simple Language for Non-Technical Users
        # ==============================================================
        
        st.markdown("---")
        
        # Calculate key numbers
        daily_mean = results['mean_forecast'] / params['time_horizon']
        daily_lower_90 = results['percentiles']['P10'] / params['time_horizon']
        daily_upper_90 = results['percentiles']['P90'] / params['time_horizon']
        total_lower_90 = results['percentiles']['P10']
        total_upper_90 = results['percentiles']['P90']
        
        # Create summary box using Streamlit components
        st.info(f"""
### 🎯 {current_lang['bottom_line']} {params['time_horizon']} {current_lang['days']}:

**📊 {current_lang['expected_daily']}:** {current_lang['around']} **{daily_mean:.0f} {current_lang['units_per_day']}**  
**📦 {current_lang['total_demand']}:** {current_lang['around']} **{results['mean_forecast']:,.0f} {current_lang['units']}** ({current_lang['over']} {params['time_horizon']} {current_lang['days']})

---

### 🎲 {current_lang['how_confident']}

✅ {current_lang['chance']} **90%** {current_lang['daily_demand_between']} **{daily_lower_90:.0f} {current_lang['and']} {daily_upper_90:.0f} {current_lang['units']}**  
✅ {current_lang['chance']} **90%** {current_lang['total_demand_between']} **{total_lower_90:,.0f} {current_lang['and']} {total_upper_90:,.0f} {current_lang['units']}**

---

### 💡 {current_lang['what_to_do']}

**{current_lang['conservative']}:** {current_lang['plan_for']} **{results['percentiles']['P75']:,.0f} {current_lang['total_units']}** ({current_lang['covers']} 75% {current_lang['of_scenarios']})  
**{current_lang['balanced']}:** {current_lang['plan_for']} **{results['mean_forecast']:,.0f} {current_lang['total_units']}** ({current_lang['average_case']})  
**{current_lang['safe']}:** {current_lang['plan_for']} **{results['percentiles']['P90']:,.0f} {current_lang['total_units']}** ({current_lang['covers']} 90% {current_lang['of_scenarios']})
        """)
        
        st.markdown("---")
        
        # Key Forecast Metrics
        st.subheader(f"🎯 {current_lang['detailed_numbers']}")
        
        # Toggle between Total and Daily view
        view_type = st.radio(
            current_lang['view_forecast_as'],
            options=[f"📊 {current_lang['total_demand_sum']}", f"📅 {current_lang['daily_average_demand']}", f"📈 {current_lang['both']}"],
            index=2,
            horizontal=True,
            help="Choose how to display forecast results"
        )
        
        # Calculate daily metrics
        daily_mean = results['mean_forecast'] / params['time_horizon']
        daily_median = results['median_forecast'] / params['time_horizon']
        daily_lower = results['lower_bound'] / params['time_horizon']
        daily_upper = results['upper_bound'] / params['time_horizon']
        
        if current_lang['total_demand_sum'] in view_type or current_lang['both'] in view_type:
            st.markdown(f"#### {current_lang['total_over']} {params['time_horizon']} {current_lang['days']}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    current_lang['mean_total'],
                    f"{results['mean_forecast']:,.0f}",
                    help=f"Average total demand over {params['time_horizon']} days"
                )
            with col2:
                st.metric(
                    current_lang['median_total'],
                    f"{results['median_forecast']:,.0f}",
                    help="Median (50th percentile) forecast"
                )
            with col3:
                st.metric(
                    f"{params['confidence_level']}% {current_lang['lower']}",
                    f"{results['lower_bound']:,.0f}",
                    help="Conservative estimate - total"
                )
            with col4:
                st.metric(
                    f"{params['confidence_level']}% {current_lang['upper']}",
                    f"{results['upper_bound']:,.0f}",
                    help="Optimistic estimate - total"
                )
        
        if current_lang['daily_average_demand'] in view_type or current_lang['both'] in view_type:
            if current_lang['both'] in view_type:
                st.markdown("---")
            st.markdown(f"#### {current_lang['daily_avg_per_day']} {params['time_horizon']} {current_lang['days']})")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    current_lang['mean_daily'],
                    f"{daily_mean:.1f}",
                    help="Average daily demand forecast"
                )
            with col2:
                st.metric(
                    current_lang['median_daily'],
                    f"{daily_median:.1f}",
                    help="Median daily demand"
                )
            with col3:
                st.metric(
                    f"{params['confidence_level']}% {current_lang['lower']}",
                    f"{daily_lower:.1f}",
                    help="Conservative daily estimate"
                )
            with col4:
                st.metric(
                    f"{params['confidence_level']}% {current_lang['upper']}",
                    f"{daily_upper:.1f}",
                    help="Optimistic daily estimate"
                )
            
            # Comparison with historical
            hist_mean = df['Demand'].mean()
            daily_diff = daily_mean - hist_mean
            daily_diff_pct = (daily_diff / hist_mean * 100) if hist_mean > 0 else 0
            
            st.info(f"""
            📊 **{current_lang['comparison_historical']}**
            - {current_lang['historical_daily_avg']}: {hist_mean:.1f}
            - {current_lang['forecasted_daily_avg']}: {daily_mean:.1f}
            - {current_lang['difference']}: {daily_diff:+.1f} ({daily_diff_pct:+.1f}%)
            """)
        
        # Usage guidance
        with st.expander(f"💡 {current_lang['which_view']}"):
            st.markdown(f"""
            ### When to use Total Demand:
            - 📦 **Inventory planning** - How much total stock to order
            - 💰 **Budget planning** - Total revenue/costs for the period
            - 📊 **Capacity planning** - Total resources needed
            - 📝 **Contract negotiations** - Total commitment amounts
            
            ### When to use Daily Average:
            - 👥 **Staffing decisions** - How many staff per day
            - 🚚 **Daily operations** - Daily delivery requirements
            - 📈 **Performance tracking** - Compare against daily KPIs
            - 🔄 **Production scheduling** - Daily production targets
            
            ### Recommended Horizon by Use Case:
            - **Operational planning**: 7-14 days (short-term decisions)
            - **Tactical planning**: 30-60 days (monthly planning)
            - **Strategic planning**: 90-365 days (quarterly/annual)
            - **Safety stock**: 7-30 days (depending on lead time)
            
            **Your current setting**: {params['time_horizon']} days
            """)
        
        # Risk Analysis
        st.subheader(f"⚠️ {current_lang['planning_scenarios']}")
        
        # Create a more user-friendly version
        scenarios_df = pd.DataFrame({
            current_lang['scenario']: [
                current_lang['minimum_expected'],
                current_lang['low_side'],
                current_lang['most_likely'],
                current_lang['high_side'],
                current_lang['safe_planning'],
                current_lang['very_safe'],
                current_lang['emergency_buffer']
            ],
            current_lang['confidence_level_col']: [
                current_lang['only_10_lower'],
                current_lang['only_25_lower'],
                current_lang['fifty_fifty'],
                current_lang['only_75_lower'],
                current_lang['only_90_lower'],
                current_lang['only_95_lower'],
                current_lang['only_99_lower']
            ],
            current_lang['total_units_needed']: [f"{v:,.0f}" for v in results['percentiles'].values()],
            current_lang['per_day']: [f"{v/params['time_horizon']:.0f}" for v in results['percentiles'].values()]
        })
        
        st.dataframe(scenarios_df, hide_index=True, use_container_width=True)
        
        # Visualizations
        st.subheader(f"📈 {current_lang['forecast_distribution']}")
        
        # Toggle for chart view
        chart_view = st.radio(
            current_lang['show_distribution_as'],
            options=[current_lang['total_demand'], current_lang['daily_average_demand']],
            index=0,
            horizontal=True,
            key="chart_view"
        )
        
        # Adjust data based on view selection
        if chart_view == current_lang['daily_average_demand']:
            chart_data = results['total_demands'] / params['time_horizon']
            chart_mean = results['mean_forecast'] / params['time_horizon']
            chart_lower = results['lower_bound'] / params['time_horizon']
            chart_upper = results['upper_bound'] / params['time_horizon']
            chart_label = current_lang['daily_average_demand']
        else:
            chart_data = results['total_demands']
            chart_mean = results['mean_forecast']
            chart_lower = results['lower_bound']
            chart_upper = results['upper_bound']
            chart_label = current_lang['total_demand']
        
        tab1, tab2, tab3 = st.tabs([current_lang['distribution'], current_lang['confidence_intervals'], current_lang['simulation_paths']])
        
        with tab1:
            # Histogram with confidence intervals
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=chart_data,
                nbinsx=100,
                name='Simulated Demand',
                marker_color='#3b82f6',
                opacity=0.7
            ))
            
            # Add vertical lines for key metrics
            fig.add_vline(x=chart_mean, line_dash="dash", line_color="red",
                         annotation_text="Mean", annotation_position="top")
            fig.add_vline(x=chart_lower, line_dash="dot", line_color="green",
                         annotation_text=f"{params['confidence_level']}% Lower", annotation_position="top left")
            fig.add_vline(x=chart_upper, line_dash="dot", line_color="green",
                         annotation_text=f"{params['confidence_level']}% Upper", annotation_position="top right")
            
            if chart_view == current_lang['daily_average_demand']:
                title_text = f"{current_lang['distribution']} - {current_lang['daily_average_demand']}"
                xaxis_text = current_lang['daily_average_demand']
            else:
                title_text = f"{current_lang['distribution']} - {current_lang['total_demand']} ({current_lang['over']} {params['time_horizon']} {current_lang['days']})"
                xaxis_text = current_lang['total_demand']
            
            fig.update_layout(
                title=title_text,
                xaxis_title=xaxis_text,
                yaxis_title="Frequency",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Box plot with percentiles
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=chart_data,
                name='Forecast Distribution',
                marker_color='#3b82f6',
                boxmean='sd'
            ))
            
            if chart_view == current_lang['daily_average_demand']:
                yaxis_text = current_lang['daily_average_demand']
            else:
                yaxis_text = current_lang['total_demand']
            
            fig.update_layout(
                title="Forecast Distribution with Confidence Intervals",
                yaxis_title=yaxis_text,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Sample simulation paths
            n_paths_to_show = min(100, n_simulations)
            sample_indices = np.random.choice(n_simulations, n_paths_to_show, replace=False)
            
            fig = go.Figure()
            
            for idx in sample_indices:
                fig.add_trace(go.Scatter(
                    x=list(range(time_horizon)),
                    y=results['simulated_demands'][idx],
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.1,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add mean path
            mean_path = results['simulated_demands'].mean(axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(time_horizon)),
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"Sample Simulation Paths (showing {n_paths_to_show} out of {n_simulations:,})",
                xaxis_title="Days",
                yaxis_title="Demand",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary Table
        st.subheader(f"📋 {current_lang['detailed_summary']}")
        
        summary_data = {
            'Metric': [
                'Simulations Run',
                'Forecast Horizon (Days)',
                'Distribution Used',
                'Mean Forecast',
                'Median Forecast',
                'Std Deviation',
                f'{params["confidence_level"]}% Lower Bound',
                f'{params["confidence_level"]}% Upper Bound',
                'Forecast Range',
                'Coefficient of Variation'
            ],
            'Value': [
                f"{params['n_simulations']:,}",
                f"{params['time_horizon']}",
                results['distribution'],
                f"{results['mean_forecast']:,.2f}",
                f"{results['median_forecast']:,.2f}",
                f"{results['std_forecast']:,.2f}",
                f"{results['lower_bound']:,.2f}",
                f"{results['upper_bound']:,.2f}",
                f"{results['upper_bound'] - results['lower_bound']:,.2f}",
                f"{(results['std_forecast'] / results['mean_forecast'] * 100):.2f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        # Simulation Validation
        with st.expander(f"🔬 {current_lang['simulation_validation']}", expanded=False):
            st.markdown("#### How the Simulation Works")
            
            st.markdown(f"""
            **Step 1: Analyze Historical Data**
            - Historical Mean: {df['Demand'].mean():.2f}
            - Historical Std Dev: {df['Demand'].std():.2f}
            - Sample Size: {len(df)} records
            
            **Step 2: Generate Random Scenarios**
            - Created {params['n_simulations']:,} different possible futures
            - Each scenario forecasts {params['time_horizon']} days
            - Total random samples: {params['n_simulations'] * params['time_horizon']:,}
            
            **Step 3: Calculate Total Demand**
            - For each scenario, sum demand across all {params['time_horizon']} days
            - Result: {params['n_simulations']:,} different total demand outcomes
            
            **Step 4: Analyze Results**
            - Mean of all scenarios: {results['mean_forecast']:,.2f}
            - {params['confidence_level']}% of scenarios fall between {results['lower_bound']:,.0f} and {results['upper_bound']:,.0f}
            """)
            
            st.markdown("#### ✅ Accuracy Check")
            
            # Expected vs Actual comparison
            expected_mean = df['Demand'].mean() * params['time_horizon']
            expected_std = df['Demand'].std() * np.sqrt(params['time_horizon'])
            
            mean_diff = abs(results['mean_forecast'] - expected_mean)
            mean_diff_pct = (mean_diff / expected_mean) * 100
            
            st.markdown(f"""
            **Expected Total Demand** (simple calculation):
            - Historical Mean × Horizon = {df['Demand'].mean():.2f} × {params['time_horizon']} = {expected_mean:.2f}
            
            **Simulated Mean Forecast**: {results['mean_forecast']:.2f}
            
            **Difference**: {mean_diff:.2f} ({mean_diff_pct:.2f}%)
            
            {"🟢 Excellent match!" if mean_diff_pct < 1 else "🟡 Good approximation" if mean_diff_pct < 5 else "🟠 Check data quality"}
            
            **Expected Std Dev** (statistical theory):
            - Historical Std × √(Horizon) = {df['Demand'].std():.2f} × √{params['time_horizon']} = {expected_std:.2f}
            
            **Simulated Std Dev**: {results['std_forecast']:.2f}
            
            """)
            
            st.markdown("#### 📊 Simulation Sample Data")
            st.markdown("Here are 5 random simulation scenarios:")
            
            sample_sims = np.random.choice(params['n_simulations'], 5, replace=False)
            sample_df = pd.DataFrame({
                f'Day {i+1}': results['simulated_demands'][sample_sims, i] 
                for i in range(min(10, params['time_horizon']))
            })
            sample_df['Total'] = results['total_demands'][sample_sims]
            sample_df.index = [f'Scenario {i+1}' for i in range(5)]
            
            st.dataframe(sample_df.style.format("{:.1f}"), use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Each row represents one possible future scenario
            - The 'Total' column is what we analyze for forecasting
            - Notice how totals vary - this represents demand uncertainty
            """)
            
            # Confidence Interval Explanation
            st.markdown("#### 📏 Understanding Confidence Intervals")
            
            alpha = (100 - params['confidence_level']) / 100
            
            st.markdown(f"""
            Your {params['confidence_level']}% confidence interval [{results['lower_bound']:,.0f}, {results['upper_bound']:,.0f}] means:
            
            - **{params['confidence_level']}% of simulations** fell within this range
            - **{alpha*100:.0f}% of simulations** fell outside (split equally on both sides)
            - **Practical meaning**: You can be {params['confidence_level']}% confident that actual demand will be in this range
            
            **Planning Recommendations:**
            - **Conservative planning**: Use {results['lower_bound']:,.0f} (you'll meet demand {50 + params['confidence_level']/2:.0f}% of the time)
            - **Balanced planning**: Use {results['mean_forecast']:,.0f} (mean forecast)
            - **Safe planning**: Use {results['upper_bound']:,.0f} (covers {params['confidence_level']}% of scenarios)
            - **Very safe planning**: Use P95 ({results['percentiles']['P95']:,.0f}) or P99 ({results['percentiles']['P99']:,.0f})
            """)
        
        # Export Options
        st.subheader(f"💾 {current_lang['export_results']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export simulation results
            export_df = pd.DataFrame({
                'Simulation': range(1, len(results['total_demands']) + 1),
                'Total_Demand': results['total_demands']
            })
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=current_lang['download_simulation'],
                data=csv,
                file_name=f'monte_carlo_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Export summary
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=current_lang['download_summary'],
                data=summary_csv,
                file_name=f'monte_carlo_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col3:
            # Export scenarios
            scenarios_csv = scenarios_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=current_lang['download_risk'],
                data=scenarios_csv,
                file_name=f'risk_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )

else:
    # Welcome Screen with attractive design
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin: 2rem 0;'>
        <h2 style='color: white; margin: 0; font-size: 2rem;'>👋 {current_lang['welcome_title']}</h2>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem;'>
            {current_lang['welcome_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <h2 style='text-align: center; color: #2d3748; margin-bottom: 2rem;'>📚 {current_lang['how_to_use']}</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%; border-top: 5px solid #48bb78;'>
            <div style='text-align: center;'>
                <h3 style='color: #48bb78; font-size: 2.5rem; margin: 0;'>1️⃣</h3>
                <h3 style='color: #2d3748; margin-top: 0.5rem;'>{current_lang['step1_title']}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(current_lang['step1_content'])
    
    with col2:
        st.markdown(f"""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%; border-top: 5px solid #667eea;'>
            <div style='text-align: center;'>
                <h3 style='color: #667eea; font-size: 2.5rem; margin: 0;'>2️⃣</h3>
                <h3 style='color: #2d3748; margin-top: 0.5rem;'>{current_lang['step2_title']}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(current_lang['step2_content'])
    
    with col3:
        st.markdown(f"""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%; border-top: 5px solid #f093fb;'>
            <div style='text-align: center;'>
                <h3 style='color: #f093fb; font-size: 2.5rem; margin: 0;'>3️⃣</h3>
                <h3 style='color: #2d3748; margin-top: 0.5rem;'>{current_lang['step3_title']}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(current_lang['step3_content'])
    
    st.markdown("---")
    
    st.subheader(f"❓ {current_lang['common_questions']}")
    
    with st.expander(current_lang['faq_data_need']):
        st.markdown(current_lang['faq_data_need_content'])
    
    with st.expander(current_lang['faq_what_tells']):
        st.markdown(current_lang['faq_what_tells_content'])
    
    with st.expander(current_lang['faq_accuracy']):
        st.markdown(current_lang['faq_accuracy_content'])
    
    with st.expander(current_lang['faq_what_do']):
        st.markdown(current_lang['faq_what_do_content'])

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 3rem; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
    <h3 style='color: white; margin: 0; font-size: 1.5rem;'>📈 Demand Forecasting Tool</h3>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;'>
        Helping you make data-driven decisions with confidence
    </p>
    <p style='color: rgba(255,255,255,0.7); margin: 1rem 0 0 0; font-size: 0.85rem;'>
        Powered by Monte Carlo Simulation & Advanced Statistics
    </p>
</div>
""", unsafe_allow_html=True)

