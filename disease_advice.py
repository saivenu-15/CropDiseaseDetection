# disease_advice.py

DISEASE_ADVICE = {
    # ========== RICE DISEASES ==========
    "Rice - Bacterial Leaf Blight": {
        "cause": "Bacterial infection caused by Xanthomonas oryzae pv. oryzae. Spreads through water, infected seeds, and plant debris.",
        "symptoms": [
            "Yellowing of leaves starting from leaf tips",
            "Water-soaked lesions that turn yellow to white",
            "Drying and wilting of leaves",
            "Lesions may have wavy margins",
            "Severe cases lead to complete leaf death"
        ],
        "treatment": [
            "Use certified disease-free seeds",
            "Apply copper-based bactericides like Copper Oxychloride",
            "Spray Streptomycin sulfate (500-1000 ppm) at early stages",
            "Remove and destroy severely infected plants",
            "Avoid excessive nitrogen fertilizer application",
            "Apply balanced NPK fertilizers"
        ],
        "prevention": [
            "Maintain proper field drainage to avoid waterlogging",
            "Grow resistant rice varieties (IR20, IR64, Swarna)",
            "Remove and burn infected plant debris after harvest",
            "Practice crop rotation with non-host crops",
            "Avoid working in fields when plants are wet",
            "Maintain proper spacing between plants for air circulation"
        ],
        "requirements": [
            "Copper-based fungicides (Copper Oxychloride 50% WP)",
            "Streptomycin sulfate solution",
            "Sprayer equipment",
            "Protective gloves and mask",
            "Disease-free certified seeds",
            "Balanced NPK fertilizer"
        ],
        "best_suggestion": "Early detection and immediate application of copper-based bactericides combined with proper field management practices can effectively control Bacterial Leaf Blight. Use resistant varieties for long-term prevention."
    },

    "Rice - Brown Spot": {
        "cause": "Fungal disease caused by Cochliobolus miyabeanus (Bipolaris oryzae). Favored by high humidity and nutrient deficiencies.",
        "symptoms": [
            "Small brown circular or oval spots on leaves",
            "Spots may have yellow halos",
            "Lesions enlarge and merge together",
            "Brown discoloration on grains",
            "Premature leaf senescence"
        ],
        "treatment": [
            "Apply fungicides like Propiconazole or Tebuconazole",
            "Spray Mancozeb (2g/L) at 10-15 day intervals",
            "Apply silicon-based fertilizers to strengthen plants",
            "Remove infected plant parts",
            "Ensure adequate potassium and zinc nutrition"
        ],
        "prevention": [
            "Use resistant varieties (IR36, IR64, MTU 1010)",
            "Maintain proper soil fertility, especially potassium",
            "Avoid water stress during critical growth stages",
            "Practice proper field sanitation",
            "Use balanced fertilization",
            "Avoid excessive nitrogen application"
        ],
        "requirements": [
            "Fungicides (Propiconazole 25% EC, Mancozeb 75% WP)",
            "Silicon-based fertilizers",
            "Potassium and zinc supplements",
            "Sprayer equipment",
            "Protective equipment"
        ],
        "best_suggestion": "Brown Spot is often a sign of nutrient deficiency. Focus on balanced fertilization with adequate potassium and zinc, combined with timely fungicide application for best results."
    },

    "Rice - Healthy Rice Leaf": {
        "cause": "No disease detected - plant is healthy.",
        "symptoms": [
            "Green, vibrant leaves",
            "No spots or discolorations",
            "Normal growth pattern",
            "Strong plant structure"
        ],
        "treatment": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Apply balanced fertilizers",
            "Practice good field hygiene"
        ],
        "prevention": [
            "Continue current management practices",
            "Regular field inspections",
            "Maintain proper plant spacing",
            "Use disease-free seeds",
            "Practice crop rotation",
            "Monitor for early disease signs"
        ],
        "requirements": [
            "Regular monitoring tools",
            "Balanced fertilizers",
            "Proper irrigation system"
        ],
        "best_suggestion": "Your crop appears healthy! Continue with good agricultural practices including regular monitoring, balanced nutrition, and proper irrigation to maintain plant health."
    },

    "Rice - Leaf Blast": {
        "cause": "Fungal disease caused by Magnaporthe oryzae. Most destructive rice disease, favored by high humidity and moderate temperatures.",
        "symptoms": [
            "Diamond-shaped lesions with pointed ends",
            "Gray centers with brown margins",
            "Lesions on leaves, stems, and panicles",
            "Node infection causes stem breakage",
            "Panicle blast causes complete grain loss"
        ],
        "treatment": [
            "Apply systemic fungicides like Tricyclazole (75% WP)",
            "Spray Isoprothiolane or Edifenphos",
            "Apply fungicides at tillering and booting stages",
            "Remove severely infected plants",
            "Ensure proper field drainage"
        ],
        "prevention": [
            "Plant resistant varieties (IR64, Swarna, BPT 5204)",
            "Avoid excessive nitrogen application",
            "Maintain proper plant spacing",
            "Practice crop rotation",
            "Remove infected plant debris",
            "Use certified disease-free seeds"
        ],
        "requirements": [
            "Tricyclazole 75% WP fungicide",
            "Isoprothiolane or Edifenphos",
            "Sprayer equipment",
            "Protective gear",
            "Disease-free seeds"
        ],
        "best_suggestion": "Leaf Blast requires immediate action. Apply Tricyclazole-based fungicides at the first sign of disease, especially during tillering and booting stages. Use resistant varieties for future plantings."
    },

    "Rice - Leaf scald": {
        "cause": "Fungal disease caused by Microdochium oryzae. Common in areas with high humidity and poor air circulation.",
        "symptoms": [
            "Scalded appearance on leaf tips",
            "Yellow to brown discoloration",
            "Lesions progress from tip to base",
            "Leaves become brittle and break easily",
            "Reduced grain filling"
        ],
        "treatment": [
            "Apply fungicides like Propiconazole or Azoxystrobin",
            "Spray Carbendazim (1g/L) at early stages",
            "Remove infected leaves if possible",
            "Improve field drainage",
            "Apply balanced fertilizers"
        ],
        "prevention": [
            "Use resistant varieties",
            "Maintain proper plant spacing for air circulation",
            "Avoid excessive nitrogen",
            "Practice proper field drainage",
            "Remove infected plant debris",
            "Use disease-free seeds"
        ],
        "requirements": [
            "Propiconazole or Azoxystrobin fungicides",
            "Carbendazim 50% WP",
            "Sprayer equipment",
            "Protective equipment"
        ],
        "best_suggestion": "Leaf scald can be managed effectively with timely fungicide application and proper field management. Focus on improving air circulation and drainage in affected areas."
    },

    "Rice - Sheath Blight": {
        "cause": "Fungal disease caused by Rhizoctonia solani. Favored by high humidity, dense planting, and excessive nitrogen.",
        "symptoms": [
            "Oval or irregular lesions on leaf sheaths",
            "Gray-green to tan colored lesions",
            "Lesions may have brown borders",
            "Infection spreads to upper leaves",
            "Premature plant death in severe cases"
        ],
        "treatment": [
            "Apply fungicides like Validamycin or Hexaconazole",
            "Spray Propiconazole (0.1%) at 10-day intervals",
            "Reduce plant density if possible",
            "Remove infected plant parts",
            "Improve field drainage"
        ],
        "prevention": [
            "Maintain proper plant spacing (20x15 cm)",
            "Avoid excessive nitrogen application",
            "Use resistant varieties where available",
            "Practice crop rotation",
            "Remove infected plant debris",
            "Maintain proper field drainage"
        ],
        "requirements": [
            "Validamycin or Hexaconazole fungicides",
            "Propiconazole 25% EC",
            "Sprayer equipment",
            "Protective gear"
        ],
        "best_suggestion": "Sheath Blight is best prevented through proper spacing and avoiding excessive nitrogen. If disease appears, apply Validamycin-based fungicides immediately and improve field conditions."
    },

    # ========== PULSE DISEASES ==========
    "Pulses - anthracnose": {
        "cause": "Fungal disease caused by Colletotrichum species. Spreads through infected seeds and plant debris, favored by warm, humid conditions.",
        "symptoms": [
            "Dark brown to black sunken lesions on leaves",
            "Lesions on stems causing stem breakage",
            "Brown spots on pods",
            "Seed discoloration and shriveling",
            "Premature defoliation"
        ],
        "treatment": [
            "Apply fungicides like Carbendazim or Thiophanate-methyl",
            "Spray Mancozeb (2g/L) at 10-day intervals",
            "Use seed treatment with fungicides",
            "Remove and destroy infected plants",
            "Apply copper-based fungicides"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Practice crop rotation with non-legume crops",
            "Remove and burn crop debris",
            "Maintain proper plant spacing",
            "Avoid overhead irrigation",
            "Use resistant varieties"
        ],
        "requirements": [
            "Carbendazim 50% WP or Thiophanate-methyl",
            "Mancozeb 75% WP",
            "Copper-based fungicides",
            "Seed treatment fungicides",
            "Sprayer equipment"
        ],
        "best_suggestion": "Anthracnose can cause significant yield loss. Start with seed treatment, use resistant varieties, and apply preventive fungicides before flowering. Remove infected plants immediately."
    },

    "Pulses - DOWNY_MILDEW_LEAF": {
        "cause": "Fungal disease caused by Peronospora species. Thrives in cool, humid conditions with poor air circulation.",
        "symptoms": [
            "Yellow patches on upper leaf surface",
            "White to grayish fungal growth on lower leaf surface",
            "Downward curling of leaves",
            "Stunted plant growth",
            "Premature leaf drop"
        ],
        "treatment": [
            "Apply fungicides like Metalaxyl or Mancozeb",
            "Spray Fosetyl-Al or Propamocarb",
            "Remove severely infected leaves",
            "Improve air circulation",
            "Reduce field humidity"
        ],
        "prevention": [
            "Use resistant varieties",
            "Maintain proper plant spacing",
            "Avoid overhead irrigation",
            "Practice crop rotation",
            "Remove infected plant debris",
            "Ensure good field drainage"
        ],
        "requirements": [
            "Metalaxyl or Mancozeb fungicides",
            "Fosetyl-Al or Propamocarb",
            "Sprayer equipment",
            "Protective equipment"
        ],
        "best_suggestion": "Downy Mildew requires immediate fungicide application at first symptoms. Improve air circulation and reduce humidity through proper spacing and irrigation management."
    },

    "Pulses - FRESH_LEAF": {
        "cause": "Healthy leaf - no disease detected.",
        "symptoms": [
            "Green, healthy appearance",
            "No spots or discolorations",
            "Normal leaf texture",
            "Proper plant growth"
        ],
        "treatment": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Apply balanced fertilizers",
            "Practice good field hygiene"
        ],
        "prevention": [
            "Continue current management practices",
            "Regular field inspections",
            "Maintain proper plant spacing",
            "Use disease-free seeds",
            "Practice crop rotation"
        ],
        "requirements": [
            "Regular monitoring",
            "Balanced fertilizers",
            "Proper irrigation"
        ],
        "best_suggestion": "Your pulse crop appears healthy! Continue with good agricultural practices including regular monitoring and balanced nutrition to maintain plant health."
    },

    "Pulses - healthy": {
        "cause": "No disease detected - plant is healthy.",
        "symptoms": [
            "Vibrant green leaves",
            "No disease symptoms",
            "Normal growth pattern",
            "Strong plant structure"
        ],
        "treatment": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Apply balanced fertilizers",
            "Practice preventive measures"
        ],
        "prevention": [
            "Regular field inspections",
            "Use disease-free seeds",
            "Maintain proper spacing",
            "Practice crop rotation",
            "Monitor for early disease signs"
        ],
        "requirements": [
            "Monitoring tools",
            "Balanced fertilizers",
            "Proper irrigation"
        ],
        "best_suggestion": "Your pulse crop is healthy! Maintain good agricultural practices and continue regular monitoring to prevent disease outbreaks."
    },

    "Pulses - leaf crinckle": {
        "cause": "Viral disease caused by various viruses including Yellow Mosaic Virus. Spread by whiteflies and infected seeds.",
        "symptoms": [
            "Crinkled and distorted leaves",
            "Yellow mosaic patterns",
            "Stunted plant growth",
            "Reduced pod formation",
            "Smaller leaves than normal"
        ],
        "treatment": [
            "Control whitefly vectors with insecticides",
            "Remove and destroy infected plants immediately",
            "Apply systemic insecticides like Imidacloprid",
            "Use neem-based products for organic control",
            "No direct cure for viral infection"
        ],
        "prevention": [
            "Use virus-free certified seeds",
            "Control whitefly populations early",
            "Plant resistant varieties",
            "Remove weed hosts",
            "Practice crop rotation",
            "Use yellow sticky traps for monitoring"
        ],
        "requirements": [
            "Systemic insecticides (Imidacloprid)",
            "Neem-based products",
            "Yellow sticky traps",
            "Protective equipment"
        ],
        "best_suggestion": "Leaf crinkle is a viral disease with no direct cure. Focus on prevention through whitefly control, using virus-free seeds, and removing infected plants immediately to prevent spread."
    },

    "Pulses - LEAFMINNER_LEAF": {
        "cause": "Insect pest damage caused by leafminer larvae (Liriomyza species). Larvae tunnel inside leaves creating visible trails.",
        "symptoms": [
            "White or brown serpentine trails on leaves",
            "Blotchy mines on leaf surface",
            "Premature leaf drop",
            "Reduced photosynthesis",
            "Stunted growth in severe cases"
        ],
        "treatment": [
            "Apply insecticides like Abamectin or Spinosad",
            "Use Neem oil or Azadirachtin",
            "Remove and destroy heavily infested leaves",
            "Apply systemic insecticides",
            "Use biological control agents"
        ],
        "prevention": [
            "Monitor fields regularly for early detection",
            "Use yellow sticky traps",
            "Practice crop rotation",
            "Remove weed hosts",
            "Maintain field hygiene",
            "Use resistant varieties where available"
        ],
        "requirements": [
            "Abamectin or Spinosad insecticides",
            "Neem oil products",
            "Yellow sticky traps",
            "Sprayer equipment",
            "Protective gear"
        ],
        "best_suggestion": "Leafminer damage can be controlled with timely insecticide application. Start monitoring early and apply treatments when mines first appear. Remove heavily infested leaves."
    },

    "Pulses - POWDER_MILDEW_LEAF": {
        "cause": "Fungal disease caused by Erysiphe or Sphaerotheca species. Favored by dry conditions with high humidity at night.",
        "symptoms": [
            "White powdery fungal growth on leaves",
            "Powdery coating on upper leaf surface",
            "Yellowing and curling of leaves",
            "Premature defoliation",
            "Reduced pod formation"
        ],
        "treatment": [
            "Apply fungicides like Sulfur or Dinocap",
            "Spray Tebuconazole or Propiconazole",
            "Use neem oil as organic alternative",
            "Remove severely infected leaves",
            "Apply at 7-10 day intervals"
        ],
        "prevention": [
            "Use resistant varieties",
            "Maintain proper plant spacing",
            "Avoid excessive nitrogen",
            "Practice crop rotation",
            "Remove infected plant debris",
            "Ensure good air circulation"
        ],
        "requirements": [
            "Sulfur-based fungicides or Dinocap",
            "Tebuconazole or Propiconazole",
            "Neem oil (for organic farming)",
            "Sprayer equipment"
        ],
        "best_suggestion": "Powdery Mildew can be effectively controlled with sulfur-based fungicides or Tebuconazole. Apply treatments early in the morning and ensure good coverage. Improve air circulation."
    },

    "Pulses - powdery mildew": {
        "cause": "Fungal disease caused by Erysiphe polygoni. Common in warm, dry conditions with high relative humidity.",
        "symptoms": [
            "White to grayish powdery coating on leaves",
            "Powdery growth on stems and pods",
            "Yellowing and premature leaf drop",
            "Reduced photosynthesis",
            "Poor pod development"
        ],
        "treatment": [
            "Apply sulfur-based fungicides",
            "Spray Tebuconazole (0.1%) or Propiconazole",
            "Use neem oil extract",
            "Remove infected plant parts",
            "Apply fungicides at 7-day intervals"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Maintain adequate spacing",
            "Avoid excessive nitrogen fertilization",
            "Practice crop rotation",
            "Remove crop residues",
            "Monitor humidity levels"
        ],
        "requirements": [
            "Sulfur fungicides (80% WP)",
            "Tebuconazole 25% EC",
            "Neem oil",
            "Sprayer equipment"
        ],
        "best_suggestion": "Powdery Mildew responds well to sulfur-based treatments. Apply fungicides preventively before disease appears, especially during flowering. Maintain proper spacing for air circulation."
    },

    "Pulses - yellow mosaic": {
        "cause": "Viral disease caused by Mungbean Yellow Mosaic Virus (MYMV). Transmitted by whiteflies.",
        "symptoms": [
            "Yellow mosaic patterns on leaves",
            "Alternating yellow and green patches",
            "Stunted plant growth",
            "Reduced pod formation",
            "Smaller leaves"
        ],
        "treatment": [
            "Control whitefly vectors with insecticides",
            "Remove infected plants immediately",
            "Apply systemic insecticides like Acetamiprid",
            "Use neem-based products",
            "No direct cure for virus"
        ],
        "prevention": [
            "Use virus-free certified seeds",
            "Control whitefly populations early",
            "Plant resistant/tolerant varieties",
            "Remove weed hosts",
            "Use yellow sticky traps",
            "Practice crop rotation"
        ],
        "requirements": [
            "Systemic insecticides (Acetamiprid, Imidacloprid)",
            "Neem products",
            "Yellow sticky traps",
            "Protective equipment"
        ],
        "best_suggestion": "Yellow Mosaic is a viral disease spread by whiteflies. Prevention is key - use virus-free seeds, control whiteflies early, and remove infected plants immediately. Resistant varieties offer best protection."
    },

    "Pulses - Downy Mildew": {
        "cause": "Fungal disease caused by Peronospora viciae or similar species. Thrives in cool, humid conditions.",
        "symptoms": [
            "Yellow patches on upper leaf surface",
            "Grayish-white fungal growth on lower surface",
            "Downward curling of leaves",
            "Stunted growth",
            "Premature defoliation"
        ],
        "treatment": [
            "Apply Metalaxyl or Mancozeb fungicides",
            "Spray Fosetyl-Al or Propamocarb",
            "Remove infected leaves",
            "Improve field ventilation",
            "Reduce humidity"
        ],
        "prevention": [
            "Use resistant varieties",
            "Maintain proper plant spacing",
            "Avoid overhead irrigation",
            "Practice crop rotation",
            "Remove infected debris",
            "Ensure good drainage"
        ],
        "requirements": [
            "Metalaxyl-M or Mancozeb",
            "Fosetyl-Al",
            "Sprayer equipment",
            "Protective gear"
        ],
        "best_suggestion": "Downy Mildew requires immediate fungicide application. Improve air circulation through proper spacing and avoid overhead irrigation. Use resistant varieties for future plantings."
    }
}
