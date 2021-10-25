// DeepNormals-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/dnn.hpp>

#include <iostream>
#include <tuple>
#include <string>



namespace {

using namespace cv;

// thinning stuff

enum ThinningTypes {
    THINNING_ZHANGSUEN = 0,  // Thinning technique of Zhang-Suen
    THINNING_GUOHALL = 1     // Thinning technique of Guo-Hall
};

// Applies a thinning iteration to a binary image
void thinningIteration(Mat img, int iter, int thinningType) {
    Mat marker = Mat::zeros(img.size(), CV_8UC1);

    if (thinningType == THINNING_ZHANGSUEN) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) + (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) marker.at<uchar>(i, j) = 1;
            }
        }
    }
    if (thinningType == THINNING_GUOHALL) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) + ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N = N1 < N2 ? N1 : N2;
                int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

                if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0))) marker.at<uchar>(i, j) = 1;
            }
        }
    }

    img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType = THINNING_ZHANGSUEN) {
    Mat processed = input.getMat().clone();
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    Mat prev = Mat::zeros(processed.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(processed, 0, thinningType);
        thinningIteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    } while (countNonZero(diff) > 0);

    processed *= 255;

    output.assign(processed);
}

} // namespace


namespace {

auto load_linedrawing(const char* Path) {
    //print('loading' + Path)
    auto img = cv::imread(Path, cv::IMREAD_GRAYSCALE);
    cv::bitwise_not(img, img); //invert image
    cv::Mat thresh1;
    cv::threshold(img, thresh1, 24, 255, cv::THRESH_BINARY);
    return thresh1;
}

auto PrepareMultiScale(const cv::Mat& src) {
    cv::Mat img;
    src.convertTo(img, CV_32F);

    enum { size = 256 };
    cv::Mat img_pad = cv::Mat::zeros(img.rows + 2 * size, img.cols + 2 * size, CV_32FC1);//np.zeros((img.shape[0] + 2 * size, img.shape[1] + 2 * size), np.float32)
    img.copyTo(img_pad(cv::Rect(size + 1, size + 1, img.cols, img.rows))); //[size + 1:(img.shape[0] + size + 1), size + 1 : (img.shape[1] + size + 1)] = img

    //resized version of image for global view
    //img_2tmp = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
    cv::Mat img_2tmp;
    cv::resize(img, img_2tmp, {}, 0.5, 0.5, cv::INTER_LINEAR);
    //img_2 = np.zeros((img_2tmp.shape[0] + 2 * size, img_2tmp.shape[1] + 2 * size), np.float32)
    //img_2[size + 1:(img_2tmp.shape[0] + size + 1), size + 1 : (img_2tmp.shape[1] + size + 1)] = img_2tmp
    cv::Mat img_2 = cv::Mat::zeros(img_2tmp.rows + 2 * size, img_2tmp.cols + 2 * size, CV_32FC1);
    img_2tmp.copyTo(img_2(cv::Rect(size + 1, size + 1, img_2tmp.cols, img_2tmp.rows)));

    //img_4tmp = cv2.resize(img_2tmp, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
    //img_4 = np.zeros((img_4tmp.shape[0] + 2 * size, img_4tmp.shape[1] + 2 * size), np.float32)
    //img_4[size + 1:(img_4tmp.shape[0] + size + 1), size + 1 : (img_4tmp.shape[1] + size + 1)] = img_4tmp
    cv::Mat img_4tmp;
    cv::resize(img_2tmp, img_4tmp, {}, 0.5, 0.5, cv::INTER_LINEAR);
    cv::Mat img_4 = cv::Mat::zeros(img_4tmp.rows + 2 * size, img_4tmp.cols + 2 * size, CV_32FC1);
    img_4tmp.copyTo(img_4(cv::Rect(size + 1, size + 1, img_4tmp.cols, img_4tmp.rows)));

    return std::make_tuple(img_pad, img_2, img_4);
}


auto BorderHandle(int x, int size_2, int lenn) {

    int xm, Xm;

    if ((x - size_2) < 0) {
        xm = 0;
        Xm = size_2 - x;
    }
    else {
        xm = (x - size_2);
        Xm = 0;
    }

    int xM, XM;

    if ((x + size_2) > lenn) {
        xM = lenn;
        XM = size_2 + (lenn - x);
    }
    else {
        xM = x + size_2;
        XM = 2 * size_2;
    }
    return std::make_tuple(xm, xM, Xm, XM);
}


auto CropMultiScale_ZeroPadding_2(int x, int y, const cv::Mat& image, const cv::Mat& image_2, const cv::Mat& image_4, int size) {

    std::vector<cv::Mat> img_blank(3);
    for (auto& v : img_blank)
        v = cv::Mat::zeros(size, size, CV_32FC1); //np.zeros((size, size, 3), np.float32)

    auto x1 = int(x / 2) + size + 1;
    auto y1 = int(y / 2) + size + 1;
    auto x2 = int(x / 4) + size + 1;
    auto y2 = int(y / 4) + size + 1;
    x = x + size + 1;
    y = y + size + 1;
    size = int(size / 2);

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x1, size, image_2.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y1, size, image_2.rows);
        //img_blank[Ym:YM, Xm : XM, 1] = image_2[ym:yM, xm : xM]
        image_2({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[1]({ Point(Xm, Ym), Point(XM, YM) }));
    }

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x2, size, image_4.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y2, size, image_4.rows);
        //img_blank[Ym:YM, Xm : XM, 2] = image_4[ym:yM, xm : xM]
        image_4({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[2]({ Point(Xm, Ym), Point(XM, YM) }));
    }

    {
        auto[xm, xM, Xm, XM] = BorderHandle(x, size, image.cols);
        auto[ym, yM, Ym, YM] = BorderHandle(y, size, image.rows);
        //img_blank[Ym:YM, Xm : XM, 0] = image[ym:yM, xm : xM]
        image({ Point(xm, ym), Point(xM, yM) }).copyTo(img_blank[0]({ Point(Xm, Ym), Point(XM, YM) }));
    }

        //img_blank = img_blank / 127.5 - 1.0
        //return img_blank

    for (auto& v : img_blank)
    {
        v /= 127.5;
        v -= 1.0;

    }

    cv::Mat result;
    cv::merge(img_blank, result);
    
    //result /= 127.5;
    //result -= 1.0;

    return result;
}


} // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
        return 1;

    try {

        auto img = load_linedrawing(argv[1]);

        //Load Mask
        auto Mask = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

        //auto net = cv::dnn::readNetFromTensorflow("/solutions/DeepNormals/frozen_model.pb");// , "/solutions/DeepNormals/graph.pbtxt");
        //auto net = cv::dnn::readNetFromTensorflow("/solutions/DeepNormals/opt_graph.pb");// , "/solutions/DeepNormals/graph.pbtxt");
        //auto net = cv::dnn::readNetFromONNX("/solutions/DeepNormals/transformed.onnx");
        auto net = cv::dnn::readNetFromModelOptimizer("/solutions/DeepNormals/ovn/frozen_model.xml", "/solutions/DeepNormals/ovn/frozen_model.bin");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        thinning(img, img);

        //img = MaskToInput(img, Mask);

        for (int y = 0; y < Mask.rows; ++y)
            for (int x = 0; x < Mask.cols; ++x)
                if (Mask.at<uchar>(y, x))
                {
                    auto& v = img.at<uchar>(y, x);
                    if (v != 255)
                        v = 160;
                }

        //cv::imshow("img", img);
        //cv::waitKey();

        auto[img_pad, img_2, img_4] = PrepareMultiScale(img);

        //////////////////////////////////////////////////////////////////////////
        /*
        if os.path.exists('{}.meta'.format(MODEL_NAME)) :
            print('model: ' + MODEL_NAME + ' loading!')
            model.load(MODEL_NAME)
            print('model: ' + MODEL_NAME + ' loaded!')
        else:
        sys.exit('Error: ' + MODEL_NAME + ' Not Found')
        */

        int height = img.rows;
        int width = img.cols;
        int size = 256;

        const auto nb_grids = 40;

        auto ind = 0;
        // recfin = np.zeros((height + 600, width + 600, 3)).astype(float)

        cv::Mat recfin = cv::Mat::zeros(height + 600, width + 600, CV_32FC3);

            //recTrim = []
            //print('Predicting grids:')
        for (int offset = 0; offset < 256; offset += int(256 / nb_grids)) {
            //SubBatch = []
            //    Pos = []
            std::vector<cv::Mat> subBatch;
            std::vector<cv::Point> pos;
            auto index = 0;

            for (int j = 0; j < int(height / 256) + 2; ++j) {
                int y = j * 256 + offset - 128;
                for (int i = 0; i <int(width / 256) + 2; ++i) {
                    int x = i * 256 + offset - 128;
                        //#st = time.time()
                    try {
                        auto sub = CropMultiScale_ZeroPadding_2(x, y, img_pad, img_2, img_4, size);

                        //cv::imshow("sub", sub);
                        //cv::waitKey();


                        //#Sub[Sub < 0.3] = 0
                        //#et = time.time() - st
                        //#print("Cropp: " + str(et))
                        //#cv2.imshow('Sub', Sub)
                        //#cv2.waitKey(0)

                        subBatch.push_back(sub);
                        ++index;
                        pos.push_back({ x, y });
                    }
                    catch (...) {
                        throw;
                    }
                }
            }

            /*
            cv::Mat inp = cv::dnn::blobFromImages(subBatch);// , 1.0 / 255, Size(416, 416), Scalar(), true, false);
            net.setInput(inp);
            std::vector<Mat> predN;
            net.forward(predN);

            std::cout << "subBatch size: " << subBatch.size() << "; predN size: " << predN.size() << '\n';
            */
            cv::Mat rec = cv::Mat::zeros(height + 900, width + 900, CV_32FC3);

            const int off = 260;
            const int s = int(size / 2);

            for (int i = 0; i < subBatch.size(); ++i)
            {
                cv::imwrite("C:/solutions/DeepNormals/saved_cpp//input_" + std::to_string(ind + i) + ".png", subBatch[i] * 127.5 + 127.5, { IMWRITE_PNG_COMPRESSION, 9 });

                cv::Mat inp = cv::dnn::blobFromImage(subBatch[i]);

                //cv::Mat inp = cv::dnn::blobFromImage(subBatch[i], 1.0, Size(), Scalar(), true);

                net.setInput(inp);
                cv::Mat predN = net.forward();

                std::vector<cv::Mat> pred;

                cv::dnn::imagesFromBlob(predN, pred);

                for (auto &p : cv::Mat_<cv::Vec3f>(pred[0])) {
                    auto coeff = 1. / sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                    p[0] *= coeff;
                    p[1] *= coeff;
                    p[2] *= coeff;
                }

                //cv::normalize(pred[0], pred[0]);

                //cv::cvtColor(pred[0], pred[0], cv::COLOR_BGR2RGB);

                //cv::imshow("pred", pred[0]);
                //cv::waitKey();

                cv::imwrite("C:/solutions/DeepNormals/saved_cpp//output_" + std::to_string(ind + i) + ".png", pred[0] * 127.5 + 127.5, { IMWRITE_PNG_COMPRESSION, 9 });

                int x = off + pos[i].x;
                int y = off + pos[i].y;
                try {
                    rec({ Point(x - s, y - s), Point(x + s, y + s) }) += pred[0];
                }
                catch (...) {
                    throw;
                }
            }

            ++ind;

            recfin({ 0, 0, width, height }) += rec({ 260, 260, width, height });

            //S = np.array(SubBatch)
            //predN = model.predict({ 'input' : S })
            /*
                        rec = np.zeros((height + 900, width + 900, 3)).astype(float)
                        off = 260
                        s = int(size / 2)
                        ind += 1.0
                        for i in range(int(index)) :
                            x = off + Pos[i][0]
                            y = off + Pos[i][1]
                            rec[(y - s) : (y + s), (x - s) : (x + s), : ] += predN[i]
                            recfin[0:height, 0 : width] += rec[260:height + 260, 260 : width + 260]
             */
        }

        //std::vector<int> vec = { ind, ind, ind };

        std::cout << ind << '\n';

        //recfin /= cv::Scalar{ double(ind), double(ind), double(ind) };
        recfin *= (.5 / ind);
        recfin += cv::Scalar{ .5, .5, .5 };

        cv::Mat result = recfin({ 0, 0, width, height });

        cv::imshow("result", result);
        cv::waitKey();

    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
