# Step 8d — Object Detection (YOLO, R-CNN)

> **Goal:** Introduction to object detection algorithms.  
> **Tools:** Python + PyTorch

---

## 8d.1 Object Detection vs Classification

**Classification:**
- What is in the image?
- Single label per image

**Detection:**
- What AND where?
- Multiple objects per image
- Bounding boxes + labels

---

## 8d.2 Bounding Boxes

Format: `(x_min, y_min, x_max, y_max)` or `(center_x, center_y, width, height)`

---

## 8d.3 R-CNN Overview

**R-CNN = Region-based CNN**

**Process:**
1. Generate region proposals
2. Extract features for each region
3. Classify each region
4. Refine bounding boxes

**Evolution:**
- R-CNN (2014): Slow
- Fast R-CNN (2015): Faster
- Faster R-CNN (2016): End-to-end

---

## 8d.4 YOLO Overview

**YOLO = You Only Look Once**

**Key idea:**
- Single pass through network
- Predict boxes and classes directly
- Much faster than R-CNN

**Process:**
1. Divide image into grid
2. Each cell predicts boxes + classes
3. Non-maximum suppression

---

## 8d.5 YOLO vs R-CNN

**R-CNN:**
- ✅ Higher accuracy
- ✅ Better for small objects
- ❌ Slower

**YOLO:**
- ✅ Very fast (real-time)
- ✅ Simpler architecture
- ❌ Lower accuracy on small objects

---

## 8d.6 Evaluation Metrics

**mAP (mean Average Precision):**
- Primary metric for detection
- Measures accuracy of boxes and classes

**IoU (Intersection over Union):**
- Measures box overlap
- Good detection: IoU > 0.5
