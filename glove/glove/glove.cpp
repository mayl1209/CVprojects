#include <opencv2\opencv.hpp>
#include <iostream>
#include <stack>

using namespace cv;
using namespace std;

struct Marker {
	Point center = Point(0, 0);
	int scale = 0;
	int colorID = 0;
	
};

const int areaThreshold = 500;
const float chromThreshold = 0.15;
const int maxDepth = 30;
Vec3b bgr[5] = {Vec3b(), Vec3b(33,183,183), Vec3b(29,158,49), Vec3b(12,9,130), Vec3b(109,33,20)};
Point2f markerColors[5] = { Point2f(-1,-1)};
Point2f jointColors[2] = { Point2f(-1,-1)};
static Point connects[4] = { Point(1,0), Point(-1,0), Point(0, 1), Point(0,-1) };

int pixelNum = 0;

int xsum = 0, ysum = 0;
int currentColorID = 0;
Mat markMatrix;
Mat currentImage;
Vec<int, 3> votes;

vector<Marker> detectMarkers();
Marker detectPalm();
void growRegion(int x, int y, bool palm);
void searchVicinity(int x, int y, int direction);
int hasMarkerColor(Vec3b pixel,bool seed);
int hasJointColor(Vec3b pixel);

int main() {
	for (int i = 1; i < 4; i++) {
		float sum = bgr[i][0] + bgr[i][1] + bgr[i][2];
		markerColors[i].x = bgr[i][0] / sum;
		markerColors[i].y = bgr[i][1] / sum;
	}
	float sum = bgr[4][0] + bgr[4][1] + bgr[4][2];
	jointColors[1].x = bgr[4][0] / sum;
	jointColors[1].y = bgr[4][1] / sum;
	//VideoCapture v(0);
	//while (true)
	//{
	//	v.read(currentImage);
	currentImage = imread("1.jpg");
	blur(currentImage, currentImage, Size(9, 9));
	Marker palm = detectPalm();
	putText(currentImage, "palm", palm.center,
		FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
	vector<Marker> markers = detectMarkers();
	int pose = 0;
	for (int i = 0; i < markers.size(); i++) {
		if ((markers[i].center.x - palm.center.x)*(markers[i].center.x - palm.center.x) +
			(markers[i].center.y - palm.center.y)*(markers[i].center.y - palm.center.y) < palm.scale * 25
			&& markers[i].scale > palm.scale / 2) {
			pose = markers[i].colorID - 1;
			putText(currentImage, "pose " + to_string(pose), markers[i].center,
				FONT_HERSHEY_SIMPLEX, 2, Scalar(255,255, 255), 3);
		}
	}
	circle(currentImage, palm.center, sqrt(palm.scale) * 5, Scalar(255, 255, 255));
	imwrite("pose1.jpg", currentImage);
	//}
}

vector<Marker> detectMarkers()
{
	vector<Marker> vm;
	markMatrix = Mat::zeros(currentImage.size(), CV_8U);
	for (int i = 0; i < currentImage.rows; i++)
		for (int j = 0; j < currentImage.cols; j++)
		{
			if (markMatrix.at<uchar>(i, j) == 0 && (currentColorID = hasMarkerColor(currentImage.at<Vec3b>(i, j),true)
				)>1) 
			{
				xsum = ysum = pixelNum = 0;
				markMatrix.at<uchar>(i, j) = currentColorID * 40;
				growRegion(i, j, 0);
				if (pixelNum > areaThreshold) {
					Marker m;
					m.center = Point(ysum / pixelNum, xsum / pixelNum);
					m.colorID = currentColorID;
					m.scale = pixelNum;
					vm.push_back(m);
				}
			}
		}
	return vm;
}

Marker detectPalm() {
	markMatrix = Mat::zeros(currentImage.size(), CV_8U);
	Marker m;
	for (int i = 0; i < currentImage.rows; i++)
		for (int j = 0; j < currentImage.cols; j++)
		{
			if (markMatrix.at<uchar>(i, j) == 0 && (currentColorID 
				= hasMarkerColor(currentImage.at<Vec3b>(i, j), true))==1)
			{
				xsum = ysum = pixelNum = 0;
				growRegion(i, j, true);
				if (pixelNum > areaThreshold&&votes[1]>pixelNum/10) {
					m.center = Point(ysum / pixelNum, xsum / pixelNum);
					m.colorID = currentColorID;
					m.scale=pixelNum;
				}
			}
		}
	imshow("1", markMatrix);
	return m;
}

void growRegion(int x, int y, bool palm)
{

	votes[0] = votes[1] = 0;
	stack<Point> seeds;
	seeds.push(Point(x, y));
	while (!seeds.empty()) {
		Point seed = seeds.top();
		seeds.pop();
		for (int i = 0; i < 4; i++) {
			int tmpx = seed.x + connects[i].x;
			int tmpy = seed.y + connects[i].y;
			if (tmpx < 0 || tmpx >= currentImage.rows
				|| tmpy < 0 || tmpy >= currentImage.cols
				|| markMatrix.at<uchar>(tmpx, tmpy) != 0
				)continue;
			int color = hasMarkerColor(currentImage.at<Vec3b>(tmpx, tmpy), false);
			//if (color == 0)continue;
			if (color!=currentColorID) {
				if(palm)
				searchVicinity(tmpx, tmpy, i);
				//markMatrix.at<uchar>(tmpx, tmpy) = 200;
			}
			else {
				seeds.push(Point(tmpx, tmpy));
				xsum += tmpx;
				ysum += tmpy;
				pixelNum++;
				markMatrix.at<uchar>(tmpx, tmpy) = currentColorID * 40;
			}
		}
	}
}

void searchVicinity(int x, int y, int direction) {
	for (int i = 0; i < maxDepth; i++) {
		if(hasJointColor(currentImage.at<Vec3b>(x, y)))
		markMatrix.at<uchar>(x, y) = 200;
		x += connects[direction].x;
		y += connects[direction].y;
		if (x < 0 || x >= currentImage.rows
			|| y < 0 || y >= currentImage.cols
			)return;
		votes[hasJointColor(currentImage.at<Vec3b>(x, y))] ++;
	}
}

int hasMarkerColor(Vec3b pixel, bool seed)
{
	float sum = pixel[0] + pixel[1] + pixel[2];
	Point2f chrom = Point2f(pixel[0] / sum, pixel[1] / sum);
	for (int i = 1; i < sizeof(markerColors); i++) {
		if ((abs(chrom.x - markerColors[i].x) + abs(chrom.y - markerColors[i].y)) < chromThreshold - (seed ? 0.08 : 0)) 
		{
			return i;
		}
	}
	return 0;
}

int hasJointColor(Vec3b pixel)
{
	float sum = pixel[0] + pixel[1] + pixel[2];
	Point2f chrom = Point2f(pixel[0] / sum, pixel[1] / sum);
	for (int i = 1; i < 2; i++) {
		if (abs(chrom.x - jointColors[i].x) + abs(chrom.y - jointColors[i].y) < chromThreshold) {
			return i;
		}
	}
	return 0;
}

