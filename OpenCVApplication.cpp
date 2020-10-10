// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <chrono> 


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


// De aici

const int SOBEL_CORE[7] = { 1, 2, 1, -1, -2 , -1, '\0'};
const int SOBEL_CORE_SIZE = 6;

const int SOBEL_X_NEIGHBOURS_X[7] = { -1, -1, -1,  1,  1,  1, '\0'};
const int SOBEL_X_NEIGHBOURS_Y[7] = { -1,  0,  1, -1,  0,  1, '\0'};

const int SOBEL_Y_NEIGHBOURS_X[7] = { -1,  0, -1, -1,  0, -1, '\0'};
const int SOBEL_Y_NEIGHBOURS_Y[7] = { -1, -1, -1,  1,  1,  1, '\0'};

const float INTERVAL_PI_PE_8 = PI / 8;
const float INTERVAL_3_PI_PE_8 = 3 * PI / 8;
const float INTERVAL_5_PI_PE_8 = 5 * PI / 8;
const float INTERVAL_7_PI_PE_8 = 7 * PI / 8;

const float INTERVAL_MINUS_PI_PE_8 =  -PI / 8;
const float INTERVAL_MINUS_3_PI_PE_8 = -3 * PI / 8;
const float INTERVAL_MINUS_5_PI_PE_8 = -5 * PI / 8;
const float INTERVAL_MINUS_7_PI_PE_8 = -7 * PI / 8;

const int DEFAULT_NEIGHBOURS_SIZE = 9;
const int DEFAULT_NEIGHBOURS_X[10] = { -1,  0,  1, -1,  0,  1, -1,  0,  1, '\0' };
const int DEFAULT_NEIGHBOURS_Y[10] = { -1, -1, -1,  0,  0,  0,  1,  1,  1, '\0' };


int coreConvolution(const Mat img, const int x, const int y, const int* NEIGHBOURS_X, const int* NEIGHBOURS_Y, const int* CORE, const int CORE_SIZE) {
	int res = 0;
	for (int i = 0; i < CORE_SIZE; i++) {
		res += img.at<uchar>(y + NEIGHBOURS_Y[i], x + NEIGHBOURS_X[i]) * CORE[i];
	}
	return res;
}

int getDirection(float teta) {
	if ((teta > INTERVAL_3_PI_PE_8 && teta < INTERVAL_5_PI_PE_8) || 
		(teta > INTERVAL_MINUS_5_PI_PE_8 && teta < INTERVAL_MINUS_3_PI_PE_8))
		return 0;
	if ((teta > INTERVAL_PI_PE_8 && teta < INTERVAL_3_PI_PE_8) || 
		(teta > INTERVAL_MINUS_7_PI_PE_8 && teta < INTERVAL_MINUS_5_PI_PE_8))
		return 1;
	if ((teta > INTERVAL_MINUS_PI_PE_8 && teta < INTERVAL_PI_PE_8) || 
		(teta > INTERVAL_7_PI_PE_8 && teta < INTERVAL_MINUS_7_PI_PE_8))
		return 2;
	if ((teta > INTERVAL_5_PI_PE_8 && teta < INTERVAL_7_PI_PE_8) || 
		(teta > INTERVAL_MINUS_3_PI_PE_8 && teta < INTERVAL_MINUS_PI_PE_8))
		return 3;
}

void computeDirectionAndModule(const Mat img, Mat Direction, Mat Module,const int width, const int height, const int d) {

	const float f4sqrt2 = 4*sqrt(2);

	for (int y = d; y < height; y++) {
		for (int x = d; x < width; x++) {
			const int gradientX = coreConvolution(img, x, y, SOBEL_X_NEIGHBOURS_X, SOBEL_X_NEIGHBOURS_Y, SOBEL_CORE, SOBEL_CORE_SIZE);
			const int gradientY = coreConvolution(img, x, y, SOBEL_Y_NEIGHBOURS_X, SOBEL_Y_NEIGHBOURS_Y, SOBEL_CORE, SOBEL_CORE_SIZE);
			
			// 11.6 Modulul gradientului
			Module.at<uchar>(y, x) = sqrt(gradientX * gradientX + gradientY * gradientY) / f4sqrt2;

			// 11.7 Directia gradientului
			float teta = atan2((float)gradientY, (float)gradientX);
			Direction.at<uchar>(y, x) = getDirection(teta);
		}
	}
}

void nonMaximSurpression(const Mat Direction, Mat Module, const int width, const int height, const int d, int *puncteModulGradientNull) {
	// Non-maxim surpression
	for (int y = d; y < height; y++) {
		for (int x = d; x < width; x++) {
			switch (Direction.at<uchar>(y, x))
			{
			case 0: {
				if (Module.at<uchar>(y, x) < Module.at<uchar>(y - 1, x) || Module.at<uchar>(y, x) < Module.at<uchar>(y + 1, x)) {
					Module.at<uchar>(y, x) = 0;
				}
				break;
			}
			case 1: {
				if (Module.at<uchar>(y, x) < Module.at<uchar>(y - 1, x - 1) || Module.at<uchar>(y, x) < Module.at<uchar>(y + 1, x + 1)) {
					Module.at<uchar>(y, x) = 0;
				}
				break;
			}
			case 2: {
				if (Module.at<uchar>(y, x) < Module.at<uchar>(y, x - 1) || Module.at<uchar>(y, x) < Module.at<uchar>(y, x + 1)) {
					Module.at<uchar>(y, x) = 0;
				}
				break;
			}
			case 3: {
				if (Module.at<uchar>(y, x) < Module.at<uchar>(y + 1, x - 1) || Module.at<uchar>(y, x) < Module.at<uchar>(y - 1, x + 1)) {
					Module.at<uchar>(y, x) = 0;
				}
				break;
			}
			default:
				break;
			}

			if(Module.at<uchar>(y, x) == 0) puncteModulGradientNull++;
		}
	}
}

int* computeHistogram(Mat Is, int* hist) {
	int height = Is.rows, width = Is.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			hist[Is.at<uchar>(i, j)]++;
		}
	}
	return hist;
}

int computePh(int* hist, int nonMuchiePrag) {
	// ingnoring hist[0]
	int sum = 0;
	for (int i = 1; i < 256; i++) {
		sum += hist[i];
		if (sum > nonMuchiePrag) {
			return i;
		}
	}
	return -1;
}

#define WEAK 128
#define STRONG 255
void manageWeak(Mat Module, const int width, const int height, const int d) {
	int val = 0, isStrong = 0;
	std::queue <Point> que;
	for (int i = d; i < height; i++) {
		for (int j = d; j < width; j++) {
			if (Module.at<uchar>(i, j) == STRONG) {
				isStrong = 1;
				que.push(Point(i, j));
				
				while (!que.empty()) {
					Point point = que.front();
					que.pop();

					for (int i = 0; i < DEFAULT_NEIGHBOURS_SIZE; i++) {
						int varX = point.x + DEFAULT_NEIGHBOURS_X[i], varY = point.y + DEFAULT_NEIGHBOURS_Y[i];
						val = Module.at<uchar>(varX, varY);
						if (val == WEAK) {
							Module.at<uchar>(varX, varY) = STRONG;
							que.push(Point(varX, varY));
						}
					}
				}
			}
		}
	}

	for (int i = d; i < height; i++) {
		for (int j = d; j < width; j++) {
			if (Module.at<uchar>(i, j) == WEAK) {
				Module.at<uchar>(i, j) = 0;
			}
		}
	}
}

Mat manageDirections(Mat Module, Mat Directions) {
	Mat res = Mat::zeros(Module.size(), CV_8UC3);
	Scalar colorLUT[4] = { 0 };
	colorLUT[0] = Scalar(0, 0, 255); //red
	colorLUT[1] = Scalar(0, 255, 0); // green
	colorLUT[2] = Scalar(255, 0, 0); // blue
	colorLUT[3] = Scalar(0, 255, 255); // yellow
	for (int i = 0; i < Module.rows; i++) {
		for (int j = 0; j < Module.cols; j++) {
			if (Module.at<uchar>(i, j)) {
				Scalar color = colorLUT[Directions.at<uchar>(i, j)];
				res.at<Vec3b>(i, j)[0] = color[0];
				res.at<Vec3b>(i, j)[1] = color[1];
				res.at<Vec3b>(i, j)[2] = color[2];
			}
		}
	}
	return res;
}

void cannyMethod() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat sourceImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", sourceImg);

		const float p = 0.1, k = 0.4;
		const int w = 3;
		const int d = w / 2;
		const int height = sourceImg.rows - d, width = sourceImg.cols - d;
		const int prod_width_height = width * height;

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		Mat Direction = Mat::zeros(sourceImg.size(), CV_8UC1);
		Mat Module = Mat::zeros(sourceImg.size(), CV_8UC1);

		int puncteModulGradientNull = 0;
		computeDirectionAndModule(sourceImg, Direction, Module, width, height, d);
		nonMaximSurpression(Direction, Module, width, height, d, &puncteModulGradientNull);
		//imshow("Module - with non maxim surpression", Module);
		

		// Binarizare adaptiva
		int hist[256] = { 0 };
		computeHistogram(Module, hist);
		// showHistogram("Histogram", hist, 256, 200);

		// 11.8
		int puncteMuchie = p * (prod_width_height - puncteModulGradientNull);
		// 11.10
		int nonMuchie = (1.0 - p) * (prod_width_height - (float)hist[0]);

		printf("Puncte muchie: %d\n", puncteMuchie);
		printf("Puncte non-muchie: %d\n", nonMuchie);

		// PragAdaptiv,  Prag inalt
		int pH = computePh(hist, nonMuchie);
		printf("pH = %d\n", pH);


		// Extinderea muchiilor prin histereza

		// Prag coborat
		int pL = (float)k * pH;
		printf("pL = %d\n", pL);

		uchar value = 0;
		for (int y = d; y < height; y++) {
			for (int x = d; x < width; x++) {
				value = Module.at<uchar>(y, x);
				Module.at<uchar>(y, x) = value < pL ? 0 : (value > pH ? STRONG : WEAK);
			}
		}
		//imshow("New module", Module);
		
		manageWeak(Module, width, height, d);
		

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

		imshow("Managed weak", Module);
		
		Mat withDirections = manageDirections(Module, Direction);
		imshow("With directions", withDirections);
		waitKey();
	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");

		printf(" 10 - Canny method\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;

			case 10:
				cannyMethod();
				break;
		}
	}
	while (op!=0);
	return 0;
}