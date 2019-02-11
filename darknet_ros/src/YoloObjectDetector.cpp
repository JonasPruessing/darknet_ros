/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

//lib for Ros pblish

#include "ros/ros.h"
// %EndTag(ROS_HEADER)%
// %Tag(MSG_HEADER)%
#include "std_msgs/String.h"
// %EndTag(MSG_HEADER)%

#include <sstream>

// for operaiting with String Pointers

#include <cstring>

#include<stdio.h>
#include<string.h>



#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <functional>


// for Json


///Trying to make darknet fast again : sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev


#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/writer.h>
#include <jsoncpp/json/value.h>
#include <string>



//2. try




/* 1st Try
/// jONAS ADDED THOSE - directing directly to files



using namespace web::http;
using namespace web::http::client;

 //// JOnas : sudo apt-get install libcpprest-dev
 //sudo apt-get install g++ git libboost-atomic-dev libboost-thread-dev libboost-system-dev libboost-date-time-dev libboost-regex-dev libboost-filesystem-dev libboost-random-dev libboost-chrono-dev libboost-serialization-dev libwebsocketpp-dev openssl libssl-dev ninja-build
 //git clone https://github.com/Microsoft/cpprestsdk.git casablanca
//cd casablanca
//mkdir build.debug
//cd build.debug
//cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Debug
//ninja

#include </usr/include/cpprest/filestream.h>



//3rd try
using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams

//


*/





#include <iostream>

#include "restclient-cpp/restclient.h"
#include "restclient-cpp/connection.h"






// yolo object detector
#include "darknet_ros/YoloObjectDetector.h"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif
/*
typedef struct node {
    int val;
    char locaname[30];
    struct node * next;
} node_t;


node_t * head = NULL;
head = malloc(sizeof(node_t));
if (head == NULL) {
return 1;
}

head->val = 1;
head->locaname = "Listenanfang"
head->next = NULL;

node_t * head = NULL;
head = malloc(sizeof(node_t));
head->val = 1;
head->next = malloc(sizeof(node_t));
head->next->val = 2;
head->next->next = NULL;

*/
int counterVar = 0;

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback, this);
  objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);
/// Jonas added/ changed line
    JonasObjektspamer_pub_ = nodeHandle_.advertise<std_msgs::String>("JonasObjektspamer", 1000);
///

  boundingBoxesPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}


// structur for detected rooms
    struct Room
    {
        char name[30];
        int weight;

    };

//arrays for roomlist
   // char roomname[100][35];
   // int roomweight[100];




void *YoloObjectDetector::detectInThread()
{

    std::cout << initVarRooms_ << '\n';
    if (initVarRooms_ == 0 )
    {

        /*

        std::string adreRooms ="http://api.conceptnet.io/c/en/room?offset=0&limit=500";

        RestClient::init();
        RestClient::Connection* connRooms = new RestClient::Connection(adreRooms);
        RestClient::HeaderFields headersRooms;
        headersRooms["Accept"] = "application/json";
        connRooms->SetHeaders(headersRooms);
        RestClient::Response rRooms = connRooms->get("");
        Json::Value rootRooms;
        Json::Reader readerRooms;
        bool parsingSuccessful = readerRooms.parse( rRooms.body, rootRooms );     //parse process
        if ( !parsingSuccessful )
        {
            std::cout  << "Failed to parse"
                       << readerRooms.getFormattedErrorMessages();
            return 0;
        }

        Json::FastWriter fastWriterRooms;

        for (int i = 0; i < rootRooms["edges"].size(); i++){


         //   std::cout << rootRooms["edges"][i]["rel"]["label"] << '\n';
         //   std::cout << rootRooms["edges"][i]["end"]["term"] << '\n';
             if (rootRooms["edges"][i]["rel"]["label"] == "IsA"  && rootRooms["edges"][i]["end"]["term"] == "/c/en/room") {



                 std::string locationRoom;
                Json::FastWriter fastWriter;
                std::string outputRooms = fastWriter.write(rootRooms["edges"][i]["start"]["term"]);
                locationRoom = outputRooms.substr(7);  // 7 = "/c/en/
                locationRoom[locationRoom.size() - 2] = '\0'; // for deleating /n
                //std::string weight = fastWriter.write(root["edges"][i]["weight"]);


                 // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ add Room to detectedRooms List
                 std::cout << locationRoom << '\n';
                 */





        detectedRooms_.emplace("living_room", 100);
        detectedRooms_.emplace("kitchen", 100);
        detectedRooms_.emplace("bathroom", 100);
        detectedRooms_.emplace("bedroom", 100);
    //    detectedRooms_.emplace("cupboard", 100);
        detectedRooms_.emplace("dining_room", 100);
        detectedRooms_.emplace("office", 100);
        detectedRooms_.emplace("nursery", 100);
        detectedRooms_.emplace("cloakroom", 100);


        

          //  }}


        initVarRooms_ = 1;


        //  std::cout << valueRooms << '\n';

    }






  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps_);
    printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_+2) % 3];

    std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';
    std::cout << "Object :: Probability" << '\n';
    std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';
  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);



  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {


        // If wahrscheinlichkeit for detection is higher then defined border wahrscheinlichkeit then print name
if (dets[i].prob[j] > demoThresh_){

    //std::cout<< demoNames_[j] <<std::endl;

    std::string ApiObjects;
    ApiObjects = demoNames_[j];

    std::replace( ApiObjects.begin(), ApiObjects.end(), ' ', '_');

    if (detectedObjects_.find(ApiObjects) == detectedObjects_.end()) {

        detectedObjects_.emplace(ApiObjects, 1);

    } else {

        detectedObjects_[ApiObjects] = 1 + detectedObjects_.at(ApiObjects);

    }

    // trying to implement ros publisher

//ros::init(argc, argv, "talker");
//ros::NodeHandle n;

//    ros::Rate loop_rate(10);
    std_msgs::String msg;

    std::stringstream ss;
    ss << demoNames_[j];
    msg.data = ss.str();
    JonasObjektspamer_pub_.publish(msg);

    /*

    std::cout << "#######################+############################+#################################" << std::endl;
    std::string adre ="http://api.conceptnet.io/c/en/" + ss.str() + "?offset=0&limit=100";


     RestClient::init();



     RestClient::Connection* conn = new RestClient::Connection(adre);
     RestClient::HeaderFields headers;
     headers["Accept"] = "application/json";
     conn->SetHeaders(headers);

     RestClient::Response r = conn->get("");



        //RestClient::Response r = RestClient::get(adre , "application/json", "{\"foo\": \"bla\"}");
   // std::cout << r.body << std::endl;

//Added reading JsonInformation:

    std::string startid ="/c/en/" + ss.str();

    Json::Value root;
    Json::Reader reader;
    bool parsingSuccessful = reader.parse( r.body, root );     //parse process
    if ( !parsingSuccessful )
    {
        std::cout  << "Failed to parse"
                   << reader.getFormattedErrorMessages();
        return 0;
    }
    //std::cout << root["edges"].size() << std::endl;

    for (int i = 0; i < root["edges"].size(); i++){

       // if ((root["edges"][i]["rel"]["label"] == "AtLocation" || root["edges"][i]["rel"]["label"] == "RelatedTo" ) && root["edges"][i]["start"]["@id"] == startid) { +++++++++++++ alternative try but adds a lot of nonsens rooms
       if (root["edges"][i]["rel"]["label"] == "AtLocation"  && root["edges"][i]["start"]["@id"] == startid) {
          //  std_msgs::String location;

            Json::FastWriter fastWriter;
            std::string output = fastWriter.write(root["edges"][i]["end"]["term"]);
            std::string weight = fastWriter.write(root["edges"][i]["weight"]);

            std::string location = output;

            // cutting stuff away
            // char roomnameholder[35];
            location = output.substr(7);  // 7 = "/c/en/
            location[location.size() - 2] = '\0'; // for deleating /n
            //location.erase(std::remove(location.begin(), location.end(), '\n'), location.end());
            std::string locationLoca = location;
            locationLoca.erase(std::remove(locationLoca.begin(), locationLoca.end(), '\n'), locationLoca.end());
            //std::string adreLoca = "http://api.conceptnet.io/relatedness?node1=/c/en/room&node2=/c/en/" + locationLoca;
            //std::string adreLoca = "http://api.conceptnet.io/query?node=/c/en/" + locationLoca + "&other=/c/en/room";
            std::string adreLoca =  "http://api.conceptnet.io/query?other=/c/en/room&node=/c/en/" + locationLoca;


            RestClient::Connection* connLoca = new RestClient::Connection(adreLoca);
            RestClient::HeaderFields headersLoca;
            headersLoca["Accept"] = "application/json";
            connLoca->SetHeaders(headersLoca);
         //  std::cout << location << std::endl;
         //   std::cout << adreLoca << std::endl;

            RestClient::Response rLoca = connLoca->get("");
            Json::Value rootLoca;
            Json::Reader readerLoca;


            bool parsingSuccessful = readerLoca.parse( rLoca.body, rootLoca );     //parse process
            if ( !parsingSuccessful )
            {
                std::cout  << "Failed to parse"
                           << reader.getFormattedErrorMessages();
                return 0;
            }



                //Json::FastWriter fastWriter;
                Json::FastWriter fastWriterLoca;

                // ++++++++++++++++++++++++++++++++++++ for Relation +++++++++++++++++++++++++
                //   std::string outputLoca = fastWriterLoca.write(rootLoca["@id"]);
             //   std::string weightLoca = fastWriterLoca.write(rootLoca["value"]);
             //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            //   std::string outputLoca = fastWriterLoca.write(rootLoca["@id"]);
            //   std::string outputLocaEndId = fastWriterLoca.write(rootLoca["edges"][0]["end"]["@id"]);
               std::string weightLoca = fastWriterLoca.write(rootLoca["edges"][0]["weight"]);

             std::string weightLocaTest = weightLoca;
           // std::cout << weightLocaTest.size() << std::endl;
           // std::cout << weightLoca << std::endl;
                //int nLocaweight = std::stoi(weightLoca);
                if (weightLocaTest.size() != 5 ) {           // ----------------------------------------------- remove NULL - not done jet -----------------------
                if ( std::stof(weightLocaTest) >= 0.5 ) {    // ----------------------------------------------- Bypass for Location<->Room Relation
                    //     std::cout << outputLoca << std::endl;
                    //   std::cout << weightLoca << std::endl;


                  //  std::cout << locationLoca << std::endl;
                 //   std::cout << weightLoca << std::endl;
                 //   std::cout << weight << std::endl;
                    //   std::string someString = root["edges"][i]["end"]["term"];


                    // char otherString[6]; // note 6, not 5, there's one there for the null terminator


                    // strncpy(otherString, someString, 5);
                    //  otherString[5] = '\0'; // place the null terminator




                    //   const string value1 = root["edges"][i]["end"]["term"];
                    // string result1 = value1.Right(3);

                    // char* Locationterm = Location;
                    //      char* pterm = Locationterm;
                    //     Locationterm = pterm - 6;


// trying to creat a List with two column


                    // std::cout << "Ort: " << locationLoca << "Gewicht: " << root["edges"][i]["weight"] << std::endl;









                    //  roomnameholder = location;
/*

            for (int i = 0; i < 100; i++) {

              if (roomname[i] = roomnameholder)
              {
                roomweight[i] = roomweight[i] + root["edges"][i]["weight"]
              } else
              {
                for (int i = 0; i < 100; i++) {

                  if (roomname[i] = '\0')
                  {
                    roomname[i] = location;
                    roomweight[i] = root["edges"][i]["weight"];

                  }

                }

              }

            }





            ;struct Room Room;

            ;strcpy( .title, "C Programming");






            struct Room location;
            strcpy(location.name, location);
            location.weight = root["edges"][i]["weight"];










            for (int i = 0; i < output.size(); i++) {
            if (i > 6) {
                location[i-5] = output[i];
            }

            }



            std::string str2 = output.substr (3,5);     // "think"

            std::size_t pos = str.find("live");      // position of "live" in str

            std::string str3 = str.substr (pos);     // get from "live" to the end

            std::cout << str2 << ' ' << str3 << '\n';

            std::cout << location << " : Gewicht: " << root["edges"][i]["weight"]  << std::endl;


                }
            }


        }
    }



*/






    /*
        RestClient::Connection* conn = new RestClient::Connection("http://api.conceptnet.io/c/en/keyboard");
        RestClient::HeaderFields headers;
        headers["Accept"] = "application/json";
        conn->SetHeaders(headers);
        RestClient::Response r = conn->get("/get");
    */






//printf("%s: %.0f%%\n  %d  ", demoNames_[j], dets[i].prob[j]*100, nboxes);
}

      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }



    counterVar ++;


    if ( counterVar >= 3 )
    {


        /*

             std::cout << "mymap contains:\n";

             std::pair<char,int> highest = detectedRooms_.rbegin();                 // last element

             std::map<char,int>::iterator it = detectedRooms_.begin();
             do {
                 std::cout << it->first << " => " << it->second << '\n';
             } while ( detectedRooms_.value_comp()(*it++, hist) );
             counterVar = 0;





        */


        for (auto& y: detectedObjects_) {
            //  std::cout << y.first << ": " << y.second << '\n';

            for (auto& x: detectedRooms_) {
                // std::cout << x.first << ": " << x.second << '\n';

                std::string adreRelat ="http://api.conceptnet.io/relatedness?node1=/c/en/" + y.first + "&node2=/c/en/" + x.first;

                RestClient::init();
                RestClient::Connection* connRelat = new RestClient::Connection(adreRelat);
                RestClient::HeaderFields headersRelat;
                headersRelat["Accept"] = "application/json";
                connRelat->SetHeaders(headersRelat);
                RestClient::Response rRelat = connRelat->get("");
                Json::Value rootRelat;
                Json::Reader readerRelat;
                bool parsingSuccessful = readerRelat.parse( rRelat.body, rootRelat );     //parse process
                if ( !parsingSuccessful )
                {
                    std::cout  << "Failed to parse"
                               << readerRelat.getFormattedErrorMessages();
                    return 0;
                }

                Json::FastWriter fastWriterRelat;

                std::string valueRelat = fastWriterRelat.write(rootRelat["value"]);
                //  std::cout << valueRelat << '\n';


                detectedRooms_[x.first] = std::stof(valueRelat) * detectedRooms_.at(x.first) * y.second + detectedRooms_.at(x.first) ;


            }


        }












        std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';
        std::cout << "Object:: Count" << '\n';
        std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';


        for (auto& x: detectedObjects_) {
            std::cout << x.first << ": " << x.second << '\n';
        }

        std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';
        std::cout << "Room :: Score" << '\n';
        std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';
/*

      for (auto& x: detectedRooms_) {
          std::cout << x.first << ": " << x.second << '\n';
      }
      std::cout << "+++++++++++++++++++++++++++++++++++" << '\n';

      */

        // Declaring a set that will store the pairs using above comparision logic
        std::vector<std::pair<std::string, int> > setOfWords(detectedRooms_.begin(), detectedRooms_.end());
        std::sort(setOfWords.begin(), setOfWords.end(), [](const std::pair<std::string, int>& elem1 , const std::pair<std::string, int>& elem2) -> bool
        {
            return elem1.second < elem2.second;
        });

        // Iterate over a set using range base for loop
        // It will display the items in sorted order of values
        for (auto& element : setOfWords) {
            std::cout << element.first << " :: " << element.second << std::endl;
        }




        RestClient::Response rWebApi2 = RestClient::post("http://localhost:8080/WEBAPI/rest/messages", "application/json", "{\"author\":\"Jonas\",\"message\":\"blaaaaa\"}" );




        counterVar = 0;
        detectedObjects_.clear();
        setOfWords.clear();

        for (auto& z: detectedRooms_) {
            detectedRooms_[z.first] = 100;

        }




    }









  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;







  return 0;
}











    void *YoloObjectDetector::fetchInThread()
{
  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  ipl_into_image(ROS_img, buff_[buffIndex_]);
  headerBuff_[buffIndex_] = imageAndHeader.header;
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{
  show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
  int c = cvWaitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return 0;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }

  return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
  while (1) {
    displayInThread(0);
  }
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[0] = imageAndHeader.header;
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    if (fullScreen_) {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
      cvMoveWindow("YOLO V3", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  while (!demoDone_) {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      }
      publishInThread();
    } else {
      char name[256];

      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }

}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }






  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;

    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }





       return 0;

}


} /* namespace darknet_ros*/
