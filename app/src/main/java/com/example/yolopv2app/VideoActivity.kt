package com.example.yolopv2app

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.ImageView
import android.widget.Switch
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.YOLOPv2App.R
import org.opencv.android.OpenCVLoader
import java.io.InputStream

data class Yolopv2Result(
    var outputBitmap: Bitmap,
    var setOfLanesLeft : MutableList<MutableList<Pair<Int, Int>>>,
    var setOfLanesRight : MutableList<MutableList<Pair<Int, Int>>>,
    var safetyFlag: Int,
    var carFlag: Int
)

class VideoActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var inputImage: ImageView
    private lateinit var inputVideo: TextureView
    private lateinit var mediaPlayer: MediaPlayer
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private lateinit var videoSwitch: Switch
    private lateinit var objectDetectionButton: Button
    private lateinit var carWarningImage: Bitmap
    private lateinit var classes:List<String>
    lateinit var yolopv2Result: Yolopv2Result

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.video_activity)

        videoSwitch = findViewById(R.id.switch2)
        videoSwitch.setOnClickListener{
            if (videoSwitch.isChecked){
                val message = "Video Mode:ON"
                Toast.makeText(this@VideoActivity, message, Toast.LENGTH_SHORT).show()
            } else {
                val message = "Video Mode:OFF"
                if (mediaPlayer.isPlaying) {
                    mediaPlayer.stop()
                    mediaPlayer.release()
                }
                Toast.makeText(this@VideoActivity, message, Toast.LENGTH_SHORT).show()
                val intent = Intent(this, MainActivity::class.java)
                startActivity(intent)
            }
        }
        OpenCVLoader.initDebug()
        objectDetectionButton = findViewById(R.id.object_detection_button)
        inputVideo = findViewById(R.id.textureView)
        inputImage = findViewById(R.id.imageView1)
        yolopv2Result = Yolopv2Result(
            outputBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888),
            setOfLanesLeft = mutableListOf(),
            setOfLanesRight = mutableListOf(),
            safetyFlag = 0,
            carFlag = 0
        )
        val aspectRatioWidth = 16
        val aspectRatioHeight = 9
        objectDetectionButton = findViewById(R.id.object_detection_button)

        carWarningImage = BitmapFactory.decodeResource(resources, R.drawable.car_warning)
        carWarningImage = Bitmap.createScaledBitmap(carWarningImage, 120, 120, true)

        classes = readClasses();
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        inputVideo.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                val parentHeight = inputVideo.height
                val desiredWidth = (parentHeight.toFloat() / aspectRatioHeight.toFloat() * aspectRatioWidth.toFloat()).toInt()

                inputVideo.layoutParams.width = desiredWidth
                inputVideo.layoutParams.height = parentHeight

                inputVideo.requestLayout()

                mediaPlayer = MediaPlayer.create(this@VideoActivity, R.raw.test6)
                mediaPlayer.let { player ->
                    player.setSurface(Surface(surface))
                    player.setOnCompletionListener {
                        player.stop()
                        player.release()
                        val message = "Video:OFF"
                        Toast.makeText(this@VideoActivity, message, Toast.LENGTH_SHORT).show()
                    }
                }
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                // Handle surface size changes
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                mediaPlayer.stop()
                mediaPlayer.release()
                return true
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                val bitmap = inputVideo.bitmap

                if (bitmap != null) {
                    performYolopv2Detection(ortSession, bitmap, yolopv2Result.setOfLanesLeft, yolopv2Result.setOfLanesRight, yolopv2Result.safetyFlag, yolopv2Result.carFlag)
                }
            }
        }
        objectDetectionButton.setOnClickListener {
            try {
                mediaPlayer.start()
                Toast.makeText(baseContext, "Performing Yolopv2 detection!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform ObjectDetection", e)
                Toast.makeText(baseContext, "Failed to perform detection", Toast.LENGTH_SHORT)
                    .show()
            }
        }

    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: VideoResult) {
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.WHITE

        paint.textSize = 20f

        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)

        val paint1 = Paint()
        paint1.alpha = 80
        if(result.carFlag == 1){
            paint1.alpha = 255
        }
        canvas.drawBitmap(carWarningImage, 10f, mutableBitmap.height - 10f - carWarningImage.height, paint1)

        inputImage.setImageBitmap(mutableBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolopv2_192x320
        Log.d("YOLOPV2", "Yolopv2 192x320 is loaded!")
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private fun performYolopv2Detection(ortSession: OrtSession, frameBitmap: Bitmap,
                                        setOfLanesLeft: MutableList<MutableList<Pair<Int, Int>>>,
                                        setOfLanesRight: MutableList<MutableList<Pair<Int, Int>>>,
                                        safetyFlag: Int,
                                        carFlag: Int) {
        val objDetector = VideoYolopv2Detector()
        val result = objDetector.detect(frameBitmap, ortEnv, ortSession, classes, setOfLanesLeft, setOfLanesRight, safetyFlag, carFlag)
        yolopv2Result.safetyFlag = result.safetyFlag
        yolopv2Result.carFlag = result.carFlag
        updateUI(result);
    }

    companion object {
        const val TAG = "YOLOPv2Detection"
    }
}