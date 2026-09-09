# Keep native JNI methods from being stripped
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep iNAV NativeBridge
-keep class com.inav.navigation.NativeBridge { *; }
-keep class com.inav.navigation.model.** { *; }

# Keep ONNX Runtime
-keep class ai.onnxruntime.** { *; }
