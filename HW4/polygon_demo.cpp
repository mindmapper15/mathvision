#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
    m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!m_data_ready)
        putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

    drawPolygon(frame, m_data_pts, m_data_ready);
    if (m_data_ready)
    {
        // polygon area
        if (m_param.compute_area)
        {
            int area = polyArea(m_data_pts);
            char str[100];
            sprintf_s(str, 100, "Area = %d", area);
            putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
        }

        // pt in polygon
        if (m_param.check_ptInPoly)
        {
            for (int i = 0; i < (int)m_test_pts.size(); i++)
            {
                if (ptInPolygon(m_data_pts, m_test_pts[i]))
                {
                    circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), cv::FILLED);
                }
                else
                {
                    circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), cv::FILLED);
                }
            }
        }

        // homography check
        if (m_param.check_homography && m_data_pts.size() == 4)
        {
            // rect points
            int rect_sz = 100;
            vector<Point> rc_pts;
            rc_pts.push_back(Point(0, 0));
            rc_pts.push_back(Point(0, rect_sz));
            rc_pts.push_back(Point(rect_sz, rect_sz));
            rc_pts.push_back(Point(rect_sz, 0));
            rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

            // draw mapping
            char* abcd[4] = { "A", "B", "C", "D" };
            for (int i = 0; i < 4; i++)
            {
                line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
                circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), cv::FILLED);
                circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), cv::FILLED);
                putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
            }

            // check homography
            int homo_type = classifyHomography(rc_pts, m_data_pts);
            char type_str[100];
            switch (homo_type)
            {
            case NORMAL:
                sprintf_s(type_str, 100, "normal");
                break;
            case CONCAVE:
                sprintf_s(type_str, 100, "concave");
                break;
            case TWIST:
                sprintf_s(type_str, 100, "twist");
                break;
            case REFLECTION:
                sprintf_s(type_str, 100, "reflection");
                break;
            case CONCAVE_REFLECTION:
                sprintf_s(type_str, 100, "concave reflection");
               break;
            default:
                sprintf_s(type_str, 100, "unknown");
            }

            putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }

        // fit circle
        if (m_param.fit_circle)
        {
            Point2d center;
            double radius = 0;
            bool ok = fitCircle(m_data_pts, center, radius);
            if (ok)
            {
                circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
                circle(frame, center, 2, Scalar(0, 255, 0), cv::FILLED);
            }
        }
    }

    imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
    int num_of_points = vtx.size();
    double area_of_poly = 0.0;

    double x1 = (double) vtx[0].x;
    double y1 = (double) vtx[0].y;

    for(int i=1; i<(num_of_points-1); i++){
        double xi = (double) vtx[i].x;
        double yi = (double) vtx[i].y;
        double xi1 = (double) vtx[i+1].x;
        double yi1 = (double) vtx[i+1].y;

        area_of_poly += ((xi - x1)*(yi1 - y1) - (xi1 - x1)*(yi - y1)) / 2;
    }

    if (area_of_poly < 0)
        area_of_poly *= -1.0;

    return (int) area_of_poly;
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
    return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
    if (pts1.size() != 4 || pts2.size() != 4)
        return -1;

    int sign[4] = {1,1,1,1};


    for(int i=1; i<3; i++){
        Point p1 = Point((pts1[i-1].x - pts1[i].x), (pts1[i-1].y - pts1[i].y));
        Point p2 = Point((pts1[i+1].x - pts1[i].x), (pts1[i+1].y - pts1[i].y));

        Point q1 = Point((pts2[i-1].x - pts2[i].x), (pts2[i-1].y - pts2[i].y));
        Point q2 = Point((pts2[i+1].x - pts2[i].x), (pts2[i+1].y - pts2[i].y));

        int jp = (p1.x * p2.y) - (p1.y * p2.x);
        int jq = (q1.x * q2.y) - (q1.y * q2.x);

        if (jp < 0)
            sign[(i-1)*2 + 0] = -1;
        else
            sign[(i-1)*2 + 0] = 1;
        
        if (jq < 0)
            sign[(i-1)*2 + 1] = -1;
        else
            sign[(i-1)*2 + 1] = 1;
    }

    if((sign[0] * sign[1]) == (sign[2] * sign[3]))
        if((sign[0] * sign[1]) > 0)
            return NORMAL;
        else
            return REFLECTION;
    else
        if((sign[0] * sign[1]) < 0)
            return CONCAVE;
        else{
            Point p1 = Point((pts1[1].x - pts1[0].x), (pts1[1].y - pts1[0].y));
            Point p2 = Point((pts1[3].x - pts1[0].x), (pts1[3].y - pts1[0].y));

            Point q1 = Point((pts2[1].x - pts2[0].x), (pts2[1].y - pts2[0].y));
            Point q2 = Point((pts2[3].x - pts2[0].x), (pts2[3].y - pts2[0].y));

            int jp = (p1.x * p2.y) - (p1.y * p2.x);
            int jq = (q1.x * q2.y) - (q1.y * q2.x);

            if((jp * jq) < 0)
                return CONCAVE_REFLECTION;
            else
                return TWIST;
        }
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
    int n = (int)pts.size();
    if (n < 3) return false;

    return false;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), cv::FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
    }
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
    if (evt == cv::EVENT_LBUTTONDOWN)
    {
        if (!m_data_ready)
        {
            m_data_pts.push_back(Point(x, y));
        }
        else
        {
            m_test_pts.push_back(Point(x, y));
        }
        refreshWindow();
    }
    else if (evt == cv::EVENT_LBUTTONUP)
    {
    }
    else if (evt == cv::EVENT_LBUTTONDBLCLK)
    {
        m_data_ready = true;
        refreshWindow();
    }
    else if (evt == cv::EVENT_RBUTTONDBLCLK)
    {
    }
    else if (evt == cv::EVENT_MOUSEMOVE)
    {
    }
    else if (evt == cv::EVENT_RBUTTONDOWN)
    {
        m_data_pts.clear();
        m_test_pts.clear();
        m_data_ready = false;
        refreshWindow();
    }
    else if (evt == cv::EVENT_RBUTTONUP)
    {
    }
    else if (evt == cv::EVENT_MBUTTONDOWN)
    {
    }
    else if (evt == cv::EVENT_MBUTTONUP)
    {
    }

    if (flags&cv::EVENT_FLAG_LBUTTON)
    {
    }
    if (flags&cv::EVENT_FLAG_RBUTTON)
    {
    }
    if (flags&cv::EVENT_FLAG_MBUTTON)
    {
    }
    if (flags&cv::EVENT_FLAG_CTRLKEY)
    {
    }
    if (flags&cv::EVENT_FLAG_SHIFTKEY)
    {
    }
    if (flags&cv::EVENT_FLAG_ALTKEY)
    {
    }
}
