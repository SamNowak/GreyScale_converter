import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.Frame;

import java.awt.image.BufferedImage;

public class SeqCPUConv {

    public static void main(String[] args) throws Exception {
        //file destinations/locations
        String inputPath  = "src/catchilling.MP4";
        String outputPath = "C:/Users/nowak/IdeaProjects/ConvolutionComparing/output_edges.mp4";

        //grabber and dimensions
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(inputPath);
        grabber.start();
        int width  = grabber.getImageWidth();
        int height = grabber.getImageHeight();
        double fps = grabber.getFrameRate();

        //Set up the recorder
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputPath, width, height, 0);
        recorder.setVideoCodec(org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264);
        recorder.setFormat("mp4");
        recorder.setFrameRate(fps);
        recorder.start();

        //converting the frame to buffered image
        Java2DFrameConverter converter = new Java2DFrameConverter();

        Frame frame;
        int frameCount = 0;
        long startTime = System.currentTimeMillis();

        //Loop through each frame and apply the edge detection on it.
        while ((frame = grabber.grabImage()) != null) {
            BufferedImage img = converter.convert(frame);
            BufferedImage edgeImg = processFrame(img);
            Frame outFrame = converter.convert(edgeImg);
            recorder.record(outFrame);
        }

        //Cleanup!
        long totalTime = System.currentTimeMillis() - startTime;
        recorder.stop();
        grabber.stop();
        System.out.println("Done! " + totalTime + " ms.");
    }

    static BufferedImage processFrame(BufferedImage img) {
        // Convert to grayscale
        int width = img.getWidth();
        int height = img.getHeight();
        int[] pixels = img.getRGB(0, 0, width, height, null, 0, width);
        int[] grayPixels = convert_to_Grayscale(pixels, width, height);

        // Pad image
        int[] padded = padding(grayPixels, width, height);

        // Define kernels
        int[] verticalKernel = {
                1, 0, -1,
                1, 0, -1,
                1, 0, -1
        };
        int[] horizontalKernel = {
                1, 1, 1,
                0, 0, 0,
                -1, -1, -1
        };

        // Convolve
        float[] vConv = convolution(padded, verticalKernel, width, height, width + 2, 0);
        float[] hConv = convolution(padded, horizontalKernel, width, height, width + 2, 0);

        // Combine edges
        float[] combined = combining_array(vConv, hConv, width, height);

        return floatArrayToImage(combined, width, height);
    }

    public static int[] convert_to_Grayscale(int[] img, int width, int height) {
        int[] gray = new int[width * height];
        for (int i = 0; i < img.length; i++) {
            int pixel = img[i];
            int a = (pixel >> 24) & 0xff;
            int r = (pixel >> 16) & 0xff;
            int g = (pixel >> 8) & 0xff;
            int b = pixel & 0xff;
            int avg = (int)(0.299 * r + 0.587 * g + 0.114 * b);
            gray[i] = (a << 24) | (avg << 16) | (avg << 8) | avg;
        }
        return gray;
    }

    public static int[] padding(int[] img, int width, int height) {
        int padW = width + 2;
        int padH = height + 2;
        int[] pad = new int[padW * padH];
        for (int y = 0; y < padH; y++) {
            for (int x = 0; x < padW; x++) {
                int srcX = Math.max(0, Math.min(width - 1, x - 1));
                int srcY = Math.max(0, Math.min(height - 1, y - 1));
                pad[y * padW + x] = img[srcY * width + srcX];
            }
        }
        return pad;
    }

    public static float[] convolution(int[] img, int[] kernel,
                                      int width, int height,
                                      int padWidth, int kernelSum) {
        float[] result = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                result[idx] = sum(img, kernel, x + 1, y + 1, padWidth, kernelSum);
            }
        }
        return result;
    }

    public static float sum(int[] img, int[] kernel,
                            int x, int y, int padWidth, int kernelSum) {
        float s = 0;
        int k = 0;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int val = img[(y + j) * padWidth + (x + i)] & 0xff;
                s += kernel[k++] * val;
            }
        }
        return (kernelSum != 0) ? s / kernelSum : s;
    }

    public static BufferedImage floatArrayToImage(float[] arr,
                                                  int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float v : arr) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = (max - min == 0) ? 1 : (max - min);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int gray = (int)((arr[idx] - min) / range * 255);
                int rgb = (gray << 16) | (gray << 8) | gray;
                img.setRGB(x, y, rgb);
            }
        }
        return img;
    }

    public static float[] combining_array(float[] a1, float[] a2,
                                          int width, int height) {
        float[] out = new float[width * height];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float)Math.sqrt(a1[i] * a1[i] + a2[i] * a2[i]);
        }
        return out;
    }
}
