export const mockData = {
  prediction: {
    status: "DEFECT DETECTED",
    defect_type: "Hole",
    confidence: 0.96,
    pipeline_used: "Morphological + Random Forest",
    features_extracted: {
      max_area: 1542.5,
      min_eccentricity: 0.12
    },
    inference_time: "22ms"
  },
  images: {
    original: "https://placehold.co/600x400/eeeeee/666666?text=Original+Fabric",
    preprocessed: "https://placehold.co/600x400/cccccc/333333?text=Illumination+Corrected",
    mask: "https://placehold.co/600x400/000000/FFFFFF?text=Binary+Mask",
    final_overlay: "https://placehold.co/600x400/eeeeee/FF0000?text=Bounding+Box+Overlay"
  },
  ab_testing_stats: [
    { pipeline: "Morphological", model: "Random Forest", f1_score: 0.92, fps: 45 },
    { pipeline: "Morphological", model: "SVM", f1_score: 0.88, fps: 40 },
    { pipeline: "Directional Gradient", model: "Random Forest", f1_score: 0.74, fps: 60 },
    { pipeline: "Directional Gradient", model: "SVM", f1_score: 0.32, fps: 55 }
  ]
};
