## 🍔 AI Food Lab: Smart Nutrition & Quality Tracker

AI Food Lab is a real-time dashboard built with Streamlit. It identifies 10 food categories, tracks calories, and analyzes food freshness using HSV color-space logic.

![project_demo](https://github.com/user-attachments/assets/10e57bf2-f381-4b8b-949e-1654e64908f7)

 
## 🌟 Key Features

#### Deep Learning Classification:
Uses a fine-tuned EfficientNetB0 model to identify food items (Apple Pie, Sushi, Chicken Wings, etc.).

#### Freshness Scanner: 
A custom Digital Image Processing (DIP) algorithm that calculates browning ratios to detect spoilage.

#### Nutritional Insights: 
Instant visualization of Calories, Protein, Fats, and Carbs via interactive Donut Charts.

#### Sleek UI:
A clean, wide-layout dashboard optimized for desktop and mobile browsers.

## 🔬 How It Works
### 1. Classification (The Brain)
The model processes a 224×224 RGB image. After being passed through the EfficientNetB0 feature extractor and a 256-neuron dense layer, it outputs a probability distribution across 10 classes.

### 2. Quality Analysis (The Science)
Instead of just guessing, the app converts the image to the HSV (Hue, Saturation, Value) color space. It masks pixels falling within the "Brown/Oxidized" range:

Fresh: Brown pixel ratio <15%

Low Quality: Brown pixel ratio >15% (Triggers a health warning)

### 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/Sahithi987654/Food-Quality-Detection-and-Calorie-Estimation.git](https://github.com/Sahithi987654/Food-Quality-Detection-and-Calorie-Estimation.git)

# Navigate into the folder
cd food-quality-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🛠️ Technical Challenges & Solutions

### 1. Keras Model Versioning & Architecture Mismatch
> [!CAUTION]
> **The Issue:** A `ValueError` occurred during deployment because the model was trained in a legacy environment, causing the `.keras` file to fail standard loading due to layer-count discrepancies.
> 
> **The Solution:** I implemented a **Model Reconstruction Fallback**. This logic manually rebuilds the `EfficientNetB0` base, injects a custom `256-unit Dense` layer with `ReLU` activation, and applies weights using `skip_mismatch=True`. This ensures the weights are correctly aligned regardless of the environment.

### 2. Input Scaling & Normalization
> [!TIP]
> **The Issue:** Early testing resulted in low confidence scores (10-15%) or "Identity Confusion" (e.g., Chicken Wings being predicted as Sushi).
> 
> **The Solution:** Identified that manual `/255` scaling was interfering with EfficientNet’s internal requirements. I pivoted to using `tf.keras.applications.efficientnet.preprocess_input`, which handles the normalization specifically required by the B0 architecture, restoring model accuracy to 87%+.



### 3. Real-Time Quality Logic (HSV vs. RGB)
> [!NOTE]
> **The Issue:** Detecting spoilage/oxidation without a massive "rotten food" dataset.
> 
> **The Solution:** Developed a heuristic approach using **Digital Image Processing (DIP)**. By shifting from RGB to the **HSV (Hue, Saturation, Value)** color space, I was able to create a mask for "brown/oxidized" pixel ranges. 
> * **Metric:** A calculated ratio of brown pixels to total surface area.
> * **Threshold:** A $>15\%$ ratio triggers a "Low Quality" warning.
>
> * ## 📊 Supported Food Categories

The model is trained to recognize the following 10 classes with high precision:

| Category | Description |
| :--- | :--- |
| 🥧 **Apple Pie** | Traditional baked dessert |
| 🥩 **Beef Tartare** | Raw seasoned beef |
| 🥗 **Caesar Salad** | Fresh greens with dressing |
| 🍰 **Cheesecake** | Creamy dessert slice |
| 🍗 **Chicken Wings** | Fried or glazed wings |
| 🍟 **French Fries** | Crispy potato strips |
| 🍔 **Hamburger** | Classic beef patty bun |
| 🍕 **Pizza** | Sliced dough with toppings |
| 🍣 **Sushi** | Traditional Japanese rolls |
| 🧇 **Waffles** | Gridded breakfast cakes |

---

### 📋 Plain List (For Search)
`Apple Pie` • `Beef Tartare` • `Caesar Salad` • `Cheesecake` • `Chicken Wings` • `French Fries` • `Hamburger` • `Pizza` • `Sushi` • `Waffles`

## 🔮 Roadmap
- [ ] **Mobile Integration:** Develop a React Native wrapper for on-the-go scanning.
- [ ] **Expanded Dataset:** Scale from 10 to 101 food categories.
- [ ] **User Auth:** Allow users to save their daily caloric intake history.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
