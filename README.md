# Real-time Pothole Detection on Roads using Gradient-based Edge Mapping
*Google Collab Link: "https://colab.research.google.com/drive/1NKlRnQ1Hyr7B_vdCcCkwjmvoRehPjp4G?usp=sharing"* 
A lightweight, fully classical computer-vision system developed in Python and OpenCV that detects potholes in road images and video streams in real time using gradient-based edge mapping. 

Developed as part of the **CS305 – Computer Vision** course (7th Semester) at the ICFAI Foundation for Higher Education (IFHE), Hyderabad.

## Project Overview
Deep-learning detectors can accurately identify road defects, but they require heavy GPU hardware and extensive training data. This project explores a high-performance, training-free, classical alternative:
1. **ROI & Preprocessing:** Restricts processing to the lower road region of interest, normalizes uneven lighting using CLAHE, and suppresses noise with Gaussian smoothing.
2. **Gradient & Edge Mapping:** Computes image gradients via Sobel operators and extracts thin, connected edges using the Canny detector.
3. **Region Building:** Fuses edge maps and applies morphological closing and contour filling to isolate candidate regions.
4. **Candidate Validation:** Uses geometric descriptors (area, aspect ratio, solidity, extent), boundary edge density, and intensity testing to filter out shadows, lane markings, and road patches.

---

## Repository Structure
```text
├── pothole_detector.py      # Main script containing pipeline, CLI, and evaluation logic
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
