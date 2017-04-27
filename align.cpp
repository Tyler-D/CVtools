#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#define IMAGE_WIDTH_STD 90
#define IMAGE_HEIGHT_STD 90
#define LANDMARK_SIZE 16
#define LANDMARK_SIZE_DOUBLE 32

using namespace dlib;
using namespace std;

int main()
{
    try
    {

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        cv::Mat temp = cv::imread("./test.png");

        cv_image<bgr_pixel> cimg(temp);
        std::vector<rectangle> faces = detector(cimg);
        std::vector<full_object_detection> shapes;
        for (unsigned long i = 0; i < faces.size(); ++i)
            shapes.push_back(pose_model(cimg, faces[i]));

        int img_width = temp.rows;

        //Align
        //以眼睛连线的中点为旋转中心，将眼睛的连线旋转到180度水平
        cv::Point2f eyesCenter = cv::Point2f( (shapes[0].part(45).x() + shapes[0].part(36).x()) * 0.5f, (shapes[0].part(45).y() + shapes[0].part(36).y()) * 0.5f );
        double dy = (shapes[0].part(45).y() - shapes[0].part(36).y());
        double dx = (shapes[0].part(45).x() - shapes[0].part(36).x());
        double angle = atan2(dy, dx) * 180.0/CV_PI;

        cv::Mat rot_mat = cv::getRotationMatrix2D(eyesCenter, angle, 1.0);
        cv::Mat rot;
        cv::warpAffine(temp, rot, rot_mat, temp.size());

        cv::imwrite("./test_new.png", rot);
        
    }
    catch (serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}
