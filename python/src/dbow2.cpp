#include <iostream>
#include <vector>
#include "DBoW2.h"
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void getFeatures(const std::vector<std::vector<DBoW2::FORB::TDescriptor> > &training_features,
  std::vector<DBoW2::FORB::pDescriptor> &features)
{
  features.resize(0);
  
  typename std::vector<std::vector<DBoW2::FORB::TDescriptor> >::const_iterator vvit;
  typename std::vector<DBoW2::FORB::TDescriptor>::const_iterator vit;
  for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
  {
    features.reserve(features.size() + vvit->size());
    for(vit = vvit->begin(); vit != vvit->end(); ++vit)
    {
      features.push_back(&(*vit));
    }
  }
}

class Vocabulary
{
 public:
    OrbVocabulary * voc;
    Vocabulary(int k =10, int L = 5, DBoW2::WeightingType weight_type = DBoW2::TF_IDF, 
        DBoW2::ScoringType scoring = DBoW2::L1_NORM, const std::string& path = std::string() ){
        if (!path.empty()){
            voc = new OrbVocabulary(path);
        }       
        else {
            voc = new OrbVocabulary(k, L, weight_type, scoring );
        }
    }  
    ~Vocabulary(){
        delete voc;
    }

    //void create(const std::vector<std::vector<cv::Mat> > &training_features){
    void create(const std::vector<std::vector<std::vector<uint8_t>>> &training_features,
         int k = 10, int L = 5){
        std::vector<std::vector<cv::Mat> > training_features_;
        training_features_.reserve(training_features.size());
        for (auto elements : training_features){
            training_features_.push_back(std::vector<cv::Mat> ());
            std::vector<cv::Mat> &current = training_features_.back();
            current.resize(elements.size());
            for (int i = 0; i < elements.size(); i++){
                std::vector<uint8_t> element = elements[i];
                current[i] = cv::Mat(1, element.size(), CV_8U, element.data()).clone();
            }
        }
        voc->create(training_features_, k, L);
    }
    
    unsigned int size(){
        return voc->size();
    }

    bool empty(){
        return voc->empty();
    }
    
    DBoW2::BowVector transform(const std::vector<std::vector<uint8_t>> features){
        std::vector<cv::Mat> features_;
        for( auto feature : features){
            cv::Mat e = cv::Mat(1, feature.size(), CV_8U, feature.data()).clone();
            features_.push_back(e);
        }
        DBoW2::BowVector word;
        voc->transform(features_, word);
        return word;
    }

    std::tuple<DBoW2::BowVector, 
    std::map<DBoW2::NodeId, std::vector<unsigned int> >> transform_get_features(
        const std::vector<std::vector<uint8_t>> features, int levelsup){
        std::vector<cv::Mat> features_;
        for( auto feature : features){
            cv::Mat e = cv::Mat(1, feature.size(), CV_8U, feature.data()).clone();
            features_.push_back(e);
        }
        DBoW2::BowVector word;
        DBoW2::FeatureVector fv;
        voc->transform(features_, word, fv, levelsup);
        return std::make_tuple(word, fv);
    }

    double score(const DBoW2::BowVector &a, const DBoW2::BowVector &b){
        return voc->score(a, b);
    }

    DBoW2::NodeId getParentNode(DBoW2::WordId wid, int levelsup){
        return voc->getParentNode(wid, levelsup);
    }

    std::vector<DBoW2::WordId> getWordsFromNode(DBoW2::NodeId nid){
        std::vector<DBoW2::WordId> dvec;
        voc->getWordsFromNode(nid, dvec);
        return dvec;
    }
    int getBranchingFactor(){
        return voc->getBranchingFactor();
    }
    int getDepthLevels(){
        return voc->getDepthLevels();
    }
    float getEffectiveLevels(){
        return voc->getEffectiveLevels();
    }
    void save(const std::string &filename){
        voc->save(filename);
    }
    void saveToTextFile(const std::string &filename){
        voc->saveToTextFile(filename);
    }
    void load(const std::string &filename){
        voc->load(filename);
    }
    void loadFromTextFile(const std::string &filename){
        voc->loadFromTextFile(filename);
    }
    int stopWords(double minWeight){
        return voc->stopWords(minWeight);
    }
    std::string repr(){
        std::stringstream ss;
        ss << *voc;
        return ss.str();
    }
};

class Database{
    public:
        OrbDatabase *db;
        Database(Vocabulary &voc, bool use_di=true, int di_levels=0,
            const std::string& path = std::string()){
           if (!path.empty()){
               db = new OrbDatabase(path);
           }
           else{
               db = new OrbDatabase(*voc.voc, use_di, di_levels);
           }
        }
        ~Database(){
            delete db;
        }
        void setVocabulary(Vocabulary &voc, bool use_di, int di_levels=0){
            db->setVocabulary(*voc.voc, use_di, di_levels);
        }
        DBoW2::EntryId add(const std::vector<std::vector<std::uint8_t>> &features){

            std::vector<cv::Mat> features_;
            static bool here = true;
            for( auto feature: features){
                cv::Mat e = cv::Mat(1, feature.size(), CV_8U, feature.data()).clone();
                features_.emplace_back(e);
                if (here){
                    std::cout << "e " << e << std::endl;
                    here = false;
                }
            } 
            return db->add(features_, NULL, NULL);
        }
        void clear(){
            db->clear();
        }
        unsigned int size(){
            return db->size();
        }
        bool usingDirectIndex(){
            return db->usingDirectIndex();
        }
        int getDirectIndexLevels(){
            return db->getDirectIndexLevels();
        }

        std::vector<DBoW2::Result> query(const std::vector<std::vector<std::uint8_t>> &features,
            int max_results = 1, int max_id = -1){

            std::vector<cv::Mat> features_;
            for( auto feature: features){
                cv::Mat e = cv::Mat(1, feature.size(), CV_8U, feature.data()).clone();
                features_.emplace_back(e);
            }
            DBoW2::QueryResults results;
            //std::cout << "max_results : "<<max_results << " max_id "<< max_id <<std::endl;
            db->query(features_, results, max_results, max_id);
            std::vector<DBoW2::Result> res = results;
            return res;
        }
        std::map<DBoW2::NodeId, std::vector<unsigned int> > retrieveFeatures(DBoW2::EntryId id){
            DBoW2::FeatureVector fv = db->retrieveFeatures(id);
            std::map<DBoW2::NodeId, std::vector<unsigned int> > f = fv;
            return f;
        }
        void save(const std::string &filename){
            db->save(filename);
        }
        void load(const std::string &filename){
            db->load(filename);
        }
        std::string repr(){
            std::stringstream ss;
            ss << *db;
            return ss.str();
        }

};


namespace py = pybind11;

PYBIND11_MODULE(pydbow2, m) {
    py::enum_<DBoW2::LNorm>(m, "LNorm")
        .value("L1", DBoW2::L1)
        .value("L2", DBoW2::L2)
        .export_values()
        ;

    py::enum_<DBoW2::WeightingType>(m, "WeightingType")
        .value("TF_IDF", DBoW2::TF_IDF)
        .value("TF", DBoW2::TF)
        .value("IDF", DBoW2::IDF)
        .value("BINARY", DBoW2::BINARY)
        .export_values()
        ;

    py::enum_<DBoW2::ScoringType>(m, "ScoringType")
        .value("L1_NORM", DBoW2::L1_NORM)
        .value("L2_NORM", DBoW2::L2_NORM)
        .value("CHI_SQUARE", DBoW2::CHI_SQUARE)
        .value("KL", DBoW2::KL)
        .value("BHATTACHARYYA", DBoW2::BHATTACHARYYA)
        .value("DOT_PRODUCT", DBoW2::DOT_PRODUCT)
        .export_values()
        ;

    py::class_<DBoW2::BowVector>(m, "BowVector")
        .def(py::init<>())
        .def("addWeight", &DBoW2::BowVector::addWeight)
        .def("addIfNotExist", &DBoW2::BowVector::addIfNotExist )
        .def("normalize", &DBoW2::BowVector::normalize)
        .def("__len__", [](const DBoW2::BowVector &v){
            return v.size();
        }) 
        .def("__getitem__", [](const DBoW2::BowVector &v, DBoW2::WordId key){
            DBoW2::BowVector::const_iterator it = v.find(key);
            if (it == v.end()){
                throw py::key_error();
            }
            return it->second;
        })
        .def("__setitem__",[](DBoW2::BowVector &v, DBoW2::WordId key, 
        DBoW2::WordValue val){
            v[key] = val;
        })
        .def("keys", [](const DBoW2::BowVector &v){
            return py::make_key_iterator(v.begin(), v.end());
        })
        // .def("items", [](const DBoW2::BowVector &v){
        //     return py::make_item_iterator(v.begin(), v.end());
        // })
        .def("__iter__", [](const DBoW2::BowVector &v){
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0,1>())
        .def("__repr__", [](const DBoW2::BowVector &v){
            std::stringstream ss;
            ss << v;
            return ss.str();     
        })
        ;

    py::class_<DBoW2::Result>(m, "Result")
        .def(py::init<>())
        .def_readonly("Id", &DBoW2::Result::Id)
        .def_readonly("Score", &DBoW2::Result::Score)
        .def_readonly("nWords", &DBoW2::Result::nWords)
        .def_readonly("bhatScore", &DBoW2::Result::bhatScore)
        .def_readonly("chiScore", &DBoW2::Result::chiScore)
        .def_readonly("sumCommonVi", &DBoW2::Result::sumCommonVi)
        .def_readonly("sumCommonWi", &DBoW2::Result::sumCommonWi)
        .def_readonly("expectedChiScore", &DBoW2::Result::expectedChiScore)
        .def("__repr__", [](const DBoW2::Result &res){
            std::stringstream ss;
            ss << res;
            return ss.str();
        })
        ;

    py::class_<Vocabulary>(m, "Vocabulary")
        .def(py::init<int , int  , DBoW2::WeightingType, 
            DBoW2::ScoringType , const std::string& >(),
            py::arg("k")= 10, py::arg("L") = 5, py::arg("weight_type") = DBoW2::TF_IDF, 
            py::arg("scoring") = DBoW2::L1_NORM,  py::arg("path") ="" )
        .def("create", &Vocabulary::create, py::arg("training_features"),
             py::arg("k") = 10, py::arg("L") = 5)
        .def("size", &Vocabulary::size)
        .def("empty", &Vocabulary::empty)
        .def("transform", &Vocabulary::transform, py::return_value_policy::copy)
        .def("transform_get_features", &Vocabulary::transform_get_features, py::return_value_policy::copy)
        .def("score", &Vocabulary::score)
        .def("getParentNode", &Vocabulary::getParentNode)
        .def("getWordsFromNode", &Vocabulary::getWordsFromNode)
        .def("getBranchingFactor", &Vocabulary::getBranchingFactor)
        .def("getDepthLevels", &Vocabulary::getDepthLevels)
        .def("getEffectiveLevels", &Vocabulary::getEffectiveLevels)
        .def("save", &Vocabulary::save)
        .def("saveToTextFile", &Vocabulary::saveToTextFile)
        .def("load", &Vocabulary::load)
        .def("loadFromTextFile", &Vocabulary::loadFromTextFile)
        .def("stopWords", &Vocabulary::stopWords)
        .def("__repr__", &Vocabulary::repr)
        ;


    py::class_<Database>(m, "Database")
        .def(py::init<Vocabulary&, bool, int, std::string>(), py::arg("voc"),
            py::arg("use_di")=true, 
            py::arg("di_levels")=0,py::arg("path") = "")
        .def("setVocabulary", &Database::setVocabulary)
        .def("add", &Database::add )
        .def("clear", &Database::clear)
        .def("size", &Database::size)
        .def("usingDirectIndex", &Database::usingDirectIndex)
        .def("getDirectIndexLevels", &Database::getDirectIndexLevels)
        .def("query", &Database::query, py::arg("features"), py::arg("max_results")=1,
             py::arg("max_id") = -1)
        .def("retrieveFeatures", &Database::retrieveFeatures)
        .def("save", &Database::save)
        .def("load", &Database::load)
        .def("__repr__", &Database::repr)
        ;

}



