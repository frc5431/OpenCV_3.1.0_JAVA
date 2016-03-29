import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

class UFinder {
	
	static {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
	}
	
	public int 
				MINH = 62,
				MINS = 62,
				MINL = 128,
				MAXH = 87,
				MAXS = 255,
				MAXL = 238,
				BLURTIMES = 3,
				DILATETIMES = 3;
	
	public double
			 k_minSize = 15, //Minimum size for a shape to be
			 k_maxSize = 145,
			 k_minArea = 110, //Minimum area the object could have
			 k_maxArea = 900, //Maximum area the object could have
			 k_minLength = 120,//Minimum length for object to be
			 k_maxLength = 300, //Max length to contour to be
			 k_minSides = 5, //Minimum amount of sides the polygon needs
			 k_maxSides = 9, //Maximum amount of sides the polygon can have
			 k_minDepth = 41, //Minimum defect (Since it's a U we want the height as a defect)
			 k_maxDepth = 70, //Maximum for a indent in the shape to be
			 k_triggerDepth = 40, //Amount until to trigger it's a defect (Should be less than minDepth, to do anythin)
			 k_min_defects = 1, //Minimum of defects according to trigger
			 k_max_defects = 2, //Maximum of defects before it quits
			 k_obscure_depth = 1000; //If something went wrong in the calculation then calm down
	
	public boolean debuging = true;
	
	public UFinder() {
		System.out.println("Started UFinder!");
	}
	
	private double distanceCalc(double centerY) {
		return centerY;
	}
	
	
	public Object[] processImage(BufferedImage originalImage, double pixelsToCenter, double leftTrigger, double rightTrigger, double minDistance, double maxDistance) {
		//byte[] data = ((DataBufferByte) originalImage.getRaster().getDataBuffer()).getData();
		Mat original = Buff2Mat(originalImage);//Imgcodecs.imread("C:/Users/David/Desktop/middleCenter.cgi.jpg");//new Mat(originalImage.getHeight(), originalImage.getWidth(), CvType.CV_8UC3);
		//original.put(0, 0, data); 
		
		//Init_return
		Object[] toRet = {false, 0, 0, 0, 0, 0, null, 0}; //Found U, Center X, Center Y, FromCenter, Direction, Left Right, BufferedImage
		
		//Blur
		Mat blurred = new Mat();
		Imgproc.medianBlur(original, blurred, BLURTIMES);
		
		//Threshold
		Mat threshed = new Mat();
		Mat ranged = new Mat();
		Imgproc.cvtColor(blurred, threshed, Imgproc.COLOR_BGR2HLS);
		Core.inRange(threshed, new Scalar(MINH, MINL, MINS), new Scalar(MAXH, MAXL, MAXS), ranged);
		threshed.release();
		//original.release();
		blurred.release();
		//data = null;
		
		//Dilate
		Mat dilated = new Mat();
		Mat struction = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2*DILATETIMES + 1, 2*DILATETIMES+1));
		Imgproc.dilate(ranged, dilated, struction);
		ranged.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(dilated, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(original, contours, -1, new Scalar(255,255,0));
		
        for(int con = 0; con < contours.size(); con++) {
        	Mat contour = contours.get(con);
        	
        	double size = contour.size().area();
        	
        	if(size > k_minSize && size < k_maxSize) {
        		debug("Passed average size... " + String.valueOf(con));
        		
	        	double area = Imgproc.contourArea(contour);
	        	
	        	if(area > k_minArea && area < k_maxArea) {
	        		debug("Passed area... " + String.valueOf(con));
	        		
	        		MatOfPoint2f matPoint = new MatOfPoint2f(contours.get(con).toArray());
	        		double perim = Imgproc.arcLength(matPoint, true);
	        		
	        		if(perim > k_minLength && perim < k_maxLength) {
	        			debug("Passed perimeter... " + String.valueOf(con));
		        		
						MatOfPoint2f approx = new MatOfPoint2f();
		        		Imgproc.approxPolyDP(matPoint, approx, 5, true);
						approx.convertTo(contour, CvType.CV_32S);
						
						double sides = approx.size().area();
						
						if(sides > k_minSides && sides < k_maxSides) {
							debug("Passed sides... " + String.valueOf(con));
							
							MatOfInt hull = new MatOfInt();
							MatOfInt4 convDef = new MatOfInt4();
							MatOfPoint multiPoint = new MatOfPoint(contours.get(con).toArray());
							Imgproc.convexHull(multiPoint, hull, false);
							hull.convertTo(contour, CvType.CV_32S);
							Imgproc.convexityDefects(multiPoint, hull, convDef);
						
							boolean passed = true;
							int amountOfDefects = 0;
							double defectDepth = 0;
							//double finDepth = 0;
							
							debug(String.valueOf(con) + "::");
							
							if(Imgproc.isContourConvex(multiPoint)) {
								debug("		It's not convex, crap!");
								passed = false;
							} else debug("		It's a convex shape");

							
							for(int defect = 0; defect < convDef.total(); defect++) {
								defectDepth = convDef.get(defect, 0)[3] / 256.0f; //Get distance in pixels and turn to float (Diagnols need to take into effect)
								debug("		Debug_Depth: " + String.valueOf(defectDepth));
								if (defectDepth > k_triggerDepth && defectDepth < k_obscure_depth) {
									if (defectDepth < k_minDepth || defectDepth > k_maxDepth) {
										passed = false;
										debug("\n\nITEMWARNING: Defect either too small or too large\n\n");
										break;
									}
									//finDepth = defectDepth;
									amountOfDefects += 1;
								}
							}
							
							if (amountOfDefects < k_min_defects || amountOfDefects > k_max_defects) passed = false;
							
							if(passed) {
								toRet[0] = true;
								debug("		Passed defects... " + String.valueOf(con));
								debug("		Found the U");
								debug("		Processing mass of U");
								contour.convertTo(contour, CvType.CV_32F);
								Moments moments = contourMoments(multiPoint);//Imgproc.HuMoments(moments, contour);
								toRet[1] = (moments.m10 / moments.m00);
								toRet[2] = (moments.m01 / moments.m00);
								debug("		MassX = " + String.valueOf((double)toRet[1]) + "\n		MassY = " + String.valueOf((double)toRet[2]));
								toRet[3] = ((double) toRet[1]) - pixelsToCenter;
								double tempDist = distanceCalc((double) toRet[2]);
								toRet[7] = tempDist;
								toRet[4] = (tempDist > minDistance && tempDist < maxDistance) ? 0 : (tempDist < minDistance) ? 1 : 2; //Left right	
								toRet[5] = (((double) toRet[3]) > leftTrigger && ((double) toRet[3]) < rightTrigger) ? 0 : (((double) toRet[3]) < leftTrigger) ? 1 : 2; //Left right	
								Point middle = new Point((double) toRet[1], (double) toRet[2]);
								
								Point rect_points[] = { new Point(0, 0), new Point(100, 100), new Point(200, 200), new Point(300, 300)};
								
								RotatedRect rect = Imgproc.minAreaRect(matPoint);
								
								Scalar blue = new Scalar(255, 0, 0, 255);
								Scalar red = new Scalar(0, 0, 255, 255);
								Imgproc.drawContours(original, contours, con, red, 2); 
								Imgproc.circle(original, middle, 5, blue, -1);
								rect.points(rect_points);
								for (int j = 0; j < 4; j++) Imgproc.line(original, rect_points[j], rect_points[(j + 1) % 4], red, 6, 8, 0);
							}
							
							hull.release();
							convDef.release();
							multiPoint.release();
						}				
						approx.release();
	        		}
	        		matPoint.release();
	        	}
        	}
        }
        toRet[6] = Mat2Buff(original);
		//Imgcodecs.imwrite("C:/Users/David/Desktop/ok.jpg", original);
        original.release();
		dilated.release();
		
		return toRet;
	}
	
	private void debug(String toDebug) {
		if(debuging) {
			System.out.println(toDebug);
		}
	}
	
	private Mat Buff2Mat(BufferedImage image) {
		byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
		mat.put(0, 0, data);
		return mat;
	}
	
	private BufferedImage Mat2Buff(Mat mat) {
		byte[] data = new byte[mat.rows()*mat.cols()*(int)(mat.elemSize())];
		mat.get(0, 0, data);
		if (mat.channels() == 3) {
		 for (int i = 0; i < data.length; i += 3) {
		  byte temp = data[i];
		  data[i] = data[i + 2];
		  data[i + 2] = temp;
		 }
		}
		BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), BufferedImage.TYPE_3BYTE_BGR);
		image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
		return image;
	}
	
	//Since the opencv 3.1.0 has a bug, so thanks to whatisor for providing a temporary fix
	private static Moments contourMoments( MatOfPoint contour ) {
		Moments m = new Moments();
		int lpt = contour.checkVector(2);
		Point[] ptsi = contour.toArray();
	                    if( lpt == 0 )
	                            return m;

	                    double a00 = 0, a10 = 0, a01 = 0, a20 = 0, a11 = 0, a02 = 0, a30 = 0, a21 = 0, a12 = 0, a03 = 0;
	                    double xi, yi, xi2, yi2, xi_1, yi_1, xi_12, yi_12, dxy, xii_1, yii_1;


	                    {
	                            xi_1 = ptsi[lpt-1].x;
	                            yi_1 = ptsi[lpt-1].y;
	                    }

	                    xi_12 = xi_1 * xi_1;
	                    yi_12 = yi_1 * yi_1;

	                    for( int i = 0; i < lpt; i++ )
	                    {

	                            {
	                                    xi = ptsi[i].x;
	                                    yi = ptsi[i].y;
	                            }

	                            xi2 = xi * xi;
	                            yi2 = yi * yi;
	                            dxy = xi_1 * yi - xi * yi_1;
	                            xii_1 = xi_1 + xi;
	                            yii_1 = yi_1 + yi;

	                            a00 += dxy;
	                            a10 += dxy * xii_1;
	                            a01 += dxy * yii_1;
	                            a20 += dxy * (xi_1 * xii_1 + xi2);
	                            a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
	                            a02 += dxy * (yi_1 * yii_1 + yi2);
	                            a30 += dxy * xii_1 * (xi_12 + xi2);
	                            a03 += dxy * yii_1 * (yi_12 + yi2);
	                            a21 += dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 +
	                                    xi2 * (yi_1 + 3 * yi));
	                            a12 += dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 +
	                                    yi2 * (xi_1 + 3 * xi));
	                            xi_1 = xi;
	                            yi_1 = yi;
	                            xi_12 = xi2;
	                            yi_12 = yi2;
	                    }
	                    float FLT_EPSILON = 1.19209e-07f;
	                    if( Math.abs(a00) > FLT_EPSILON )
	                    {
	                            double db1_2, db1_6, db1_12, db1_24, db1_20, db1_60;

	                            if( a00 > 0 )
	                            {
	                                    db1_2 = 0.5;
	                                    db1_6 = 0.16666666666666666666666666666667;
	                                    db1_12 = 0.083333333333333333333333333333333;
	                                    db1_24 = 0.041666666666666666666666666666667;
	                                    db1_20 = 0.05;
	                                    db1_60 = 0.016666666666666666666666666666667;
	                            }
	                            else
	                            {
	                                    db1_2 = -0.5;
	                                    db1_6 = -0.16666666666666666666666666666667;
	                                    db1_12 = -0.083333333333333333333333333333333;
	                                    db1_24 = -0.041666666666666666666666666666667;
	                                    db1_20 = -0.05;
	                                    db1_60 = -0.016666666666666666666666666666667;
	                            }

	                            // spatial moments
	                            m.m00 = a00 * db1_2;
	                            m.m10 = a10 * db1_6;
	                            m.m01 = a01 * db1_6;
	                            m.m20 = a20 * db1_12;
	                            m.m11 = a11 * db1_24;
	                            m.m02 = a02 * db1_12;
	                            m.m30 = a30 * db1_20;
	                            m.m21 = a21 * db1_60;
	                            m.m12 = a12 * db1_60;
	                            m.m03 = a03 * db1_20;

	                           //m.completeState();
	                    }
	                    return m;
	            }
	
}

public class Main
{
	public static void main(String[] arg) {
		UFinder ufinder = new UFinder(); //Init class
		
		
		ufinder.debuging = true; //To flush text to terminal or not
		
		//Main Settings (Lower Hue area to Higher Hue area)
		// H should be on a bar from 0 - 180
		// S and V should be on a bar from 0 - 255
		ufinder.MINH = 62;
		ufinder.MINS = 62;
		ufinder.MINL = 128;
		
		ufinder.MAXH = 87;
		ufinder.MAXS = 255;
		ufinder.MAXL = 238;
		
		//If you want to add other settings you can, just do something like this
		//ufinder.BLURTIMES = 3; (Must be an odd number)
		
		BufferedImage toProcImage = null;
		
		Object[] returned = ufinder.processImage(toProcImage, 160, -10, 10, 80, 100); //Put in loop
		//Inputs - BufferedImage to process, pixelsToCenter, LeftPixelTrigger, RightPixelTrigger, MinimumDistance, MaximumDistance
		//Output - BufferedImage with rotated rectangle on it
		BufferedImage procImage = (BufferedImage) returned[6]; //Returned image
		boolean uFound = (boolean) returned[0]; //If the U is on the screen
		
		double centerX = (double) returned[1]; //MassCenterx position
		double centerY = (double) returned[2]; //MassCentery
		double distance = (double) returned[7]; //DistanceFromTower
		
		double fromCenterX = (double) returned[3]; //Pixels from center
		
		int forback = (int) returned[4]; // 1 = backward 0 = none 2 = forward
		int lefight = (int) returned[5]; // 1 = left 0 = none 2 = right
		
	}

}