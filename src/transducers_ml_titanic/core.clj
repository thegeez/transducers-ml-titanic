(ns transducers-ml-titanic.core
  (:require [incanter.charts :as c]
            [incanter.core :as i]
            [incanter.io :as iio]
            [incanter.stats :as stats]
            [clojure.pprint :as pprint]
            [clojure.string :as string]
            [net.cgrand.xforms :as x]
            [net.cgrand.xforms.rfs :as rf]
            [clj-ml.data :as cm-data]
            [clj-ml.classifiers :as cm-classifiers]))

;; Use incanter to read the csv. Will be converted to plain Clojure later
;; Automatically parses int, string, float etc in columns
;; This is an incanter dataset object
(def idata (try (iio/read-dataset "data/train.csv" :header true)
               (catch Exception e
                 (throw (ex-info "Be sure to put the .csv files from the kaggle titanic competition in the /data directory" {:e e})))))

;; Test data has empty values in the Survived column
(def test-idata (try (iio/read-dataset "data/test.csv" :header true)
                    (catch Exception e
                      (throw (ex-info "Be sure to put the .csv files from the kaggle titanic competition in the /data directory" {:e e})))))

(comment
  (i/col-names idata)
  ;; [:PassengerId :Survived :Pclass :Name :Sex :Age :SibSp :Parch :Ticket :Fare :Cabin :Embarked]
  (i/nrow idata)
  ;; 802
  (i/ncol idata)
  ;; 12
  )

;; Convert the incanter dataset into a plain Clojure datastructure of a vector of maps
(def data (:rows idata))

(comment
  (take 4 data)
  ({:Cabin nil, :SibSp 1, :Fare 7.25, :Embarked "S", :Sex "male", :PassengerId 1, :Ticket "A/5 21171", :Name "Braund, Mr. Owen Harris", :Survived 0, :Parch 0, :Pclass 3, :Age 22}
   {:Cabin "C85", :SibSp 1, :Fare 71.2833, :Embarked "C", :Sex "female", :PassengerId 2, :Ticket "PC 17599", :Name "Cumings, Mrs. John Bradley (Florence Briggs Thayer)", :Survived 1, :Parch 0, :Pclass 1, :Age 38}
   {:Cabin nil, :SibSp 0, :Fare 7.925, :Embarked "S", :Sex "female", :PassengerId 3, :Ticket "STON/O2. 3101282", :Name "Heikkinen, Miss. Laina", :Survived 1, :Parch 0, :Pclass 3, :Age 26}
   {:Cabin "C123", :SibSp 1, :Fare 53.1, :Embarked "S", :Sex "female", :PassengerId 4, :Ticket 113803, :Name "Futrelle, Mrs. Jacques Heath (Lily May Peel)", :Survived 1, :Parch 0, :Pclass 1, :Age 35})

  (pprint/print-table (take 4 data))
;;   | :Cabin | :SibSp |   :Fare | :Embarked |   :Sex | :PassengerId |          :Ticket |                                               :Name | :Survived | :Parch | :Pclass | :Age |
;; |--------+--------+---------+-----------+--------+--------------+------------------+-----------------------------------------------------+-----------+--------+---------+------|
;; |        |      1 |    7.25 |         S |   male |            1 |        A/5 21171 |                             Braund, Mr. Owen Harris |         0 |      0 |       3 |   22 |
;; |    C85 |      1 | 71.2833 |         C | female |            2 |         PC 17599 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) |         1 |      0 |       1 |   38 |
;; |        |      0 |   7.925 |         S | female |            3 | STON/O2. 3101282 |                              Heikkinen, Miss. Laina |         1 |      0 |       3 |   26 |
;; |   C123 |      1 |    53.1 |         S | female |            4 |           113803 |        Futrelle, Mrs. Jacques Heath (Lily May Peel) |         1 |      0 |       1 |   35 |
  )

;; There's no .info equivalent
;; Skipping ahead to the analysis per column

;; frequency transducer
(def xfrequencies (comp (x/by-key identity
                                  x/count)
                        (x/into {})))

;; Relation Pclass and Survived
(comment
  (->> data
       (into []
             (x/by-key :Pclass
                       :Survived
                       (comp x/avg
                             (map double))))
       (sort-by first <))
  #_([1 0.6296296296296297]
     [2 0.4728260869565217]
     [3 0.2423625254582485])

  ;; with avg and counts:
  (->> data
       (into []
             (x/by-key :Pclass
                       :Survived
                       (x/transjuxt {:avg (comp x/avg
                                                (map double))
                                     :count xfrequencies}))
             )
       (sort-by first <)
       )
  #_([1 {:avg 0.6296296296296297, :count {1 136, 0 80}}]
     [2 {:avg 0.4728260869565217, :count {1 87, 0 97}}]
     [3 {:avg 0.2423625254582485, :count {0 372, 1 119}}])
  )

;; Relation Sex and Survived
(defn xColumnSurvived [column]
  (x/by-key column
            :Survived
            (x/transjuxt {:avg (comp x/avg
                                     (map double))
                          :count xfrequencies})))

(comment
  (into {}
        (xColumnSurvived :Sex)
        data)
  {"male" {:avg 0.1889081455805893, :count {0 468, 1 109}},
   "female" {:avg 0.7420382165605096, :count {1 233, 0 81}}}
  )

;; Relation SibSp and Survived
(comment
  (->> (into {}
             (xColumnSurvived :SibSp)
             data)
       (sort-by key))
  ([0 {:avg 0.3453947368421053, :count {1 210, 0 398}}]
   [1 {:avg 0.5358851674641149, :count {0 97, 1 112}}]
   [2 {:avg 0.4642857142857143, :count {0 15, 1 13}}]
   [3 {:avg 0.25, :count {0 12, 1 4}}]
   [4 {:avg 0.1666666666666667, :count {0 15, 1 3}}]
   [5 {:avg 0.0, :count {0 5}}]
   [8 {:avg 0.0, :count {0 7}}])
  )

;; Relation Parch and Survived
(comment
  (->> (into {}
             (xColumnSurvived :Parch)
             data)
       (sort-by key))
  ([0 {:avg 0.3436578171091445, :count {0 445, 1 233}}]
   [1 {:avg 0.5508474576271186, :count {0 53, 1 65}}]
   [2 {:avg 0.5, :count {1 40, 0 40}}]
   [3 {:avg 0.6, :count {0 2, 1 3}}]
   [4 {:avg 0.0, :count {0 4}}]
   [5 {:avg 0.2, :count {0 4, 1 1}}]
   [6 {:avg 0.0, :count {0 1}}])
  )

;; Visualize Age per Survived
(comment
  (let [by-survived
        (into {}
              (x/by-key :Survived
                        :Age
                        ;; chart can't handle nils
                        (comp (remove nil?)
                              (x/into [])))
              data)]
    (doseq [[survived ages] by-survived]
      (-> ages
          (c/histogram
           :title (str "Age hist survived = " survived)
           :x-label "age"
           :y-label "frequency"
           :nbins 20)
          (c/set-y-range 0 60)
          i/view)))
  )

;; Visualize Age per Pclass Survived
(comment
  (let [by-pclass-survived
        (into {}
              (x/by-key (juxt :Pclass :Survived)
                        :Age
                        ;; chart can't handle nils
                        (comp (remove nil?)
                              (x/into [])))
              data)]
    (doseq [[[pclass survived] ages] by-pclass-survived]
      (-> ages
          (c/histogram
           :title (str "Age hist Pclass = " pclass " survived = " survived)
           :x-label "age"
           :y-label "frequency"
           :nbins 20)
          (c/set-y-range 0 45)
          i/view)))
  )

;; Embarked and Pclass  relation to Surived
;; (Note: the kaggle tutorial claims males have a higher survived rate when embarked in "C", this is not what is in the supplied dataset)
(comment
  (->> (into []
             (comp (x/by-key (juxt :Embarked
                                   :Sex
                                   :Pclass)
                             :Survived
                             (x/transjuxt {:count x/count
                                           :sd x/sd})
                             )
                   (map (fn [[facet stats]]
                          (assoc stats :facet facet))))
             data)
       (sort-by :facet)
       (pprint/print-table [:facet :count :sd])
       )
  ;; No standard pointplot in incanter, but here's the data:
  ;; |           :facet | :count |                 :sd |
  ;; |------------------+--------+---------------------|
  ;; | [nil "female" 1] |      2 |                 0.0 |
  ;; | ["C" "female" 1] |     43 | 0.15249857033260467 |
  ;; | ["C" "female" 2] |      7 |                 0.0 |
  ;; | ["C" "female" 3] |     23 | 0.48698475355767396 |
  ;; |   ["C" "male" 1] |     42 |   0.496795772414547 |
  ;; |   ["C" "male" 2] |     10 |  0.4216370213557839 |
  ;; |   ["C" "male" 3] |     43 | 0.42746257437545837 |
  ;; | ["Q" "female" 1] |      1 |                 0.0 |
  ;; | ["Q" "female" 2] |      2 |                 0.0 |
  ;; | ["Q" "female" 3] |     33 | 0.45226701686664544 |
  ;; |   ["Q" "male" 1] |      1 |                 0.0 |
  ;; |   ["Q" "male" 2] |      1 |                 0.0 |
  ;; |   ["Q" "male" 3] |     39 | 0.26995276239950855 |
  ;; | ["S" "female" 1] |     48 | 0.20194093652345885 |
  ;; | ["S" "female" 2] |     67 |  0.2876942444512339 |
  ;; | ["S" "female" 3] |     88 | 0.48689728436010127 |
  ;; |   ["S" "male" 1] |     79 |  0.4813968639321409 |
  ;; |   ["S" "male" 2] |     97 |   0.363438617741675 |
  ;; |   ["S" "male" 3] |    265 |  0.3350584291484042 |
  )

;; Visualize Avg Fare per Sex per Embarked Survived
(comment
  (let [tabled
        (into {}
              (x/by-key (juxt :Embarked :Survived)
                        (comp (x/by-key :Sex
                                        :Fare
                                        x/avg)
                              (x/into {})))
              data)]
    (doseq [[[embarked survived] sex+fare] tabled]
      (-> (c/bar-chart
           (keys sex+fare)
           (vals sex+fare)
           :title (str "Embarked = " embarked " survived = " survived)
           :x-label "Sex"
           :y-label "Avg Fare")
          (c/set-y-range 0 90)
          i/view)))
  )

;; Skipping creating a Title feature

;; Complete column :Age replacing nil values with avg age per class and sex
;; Visualize Age per Pclass and Sex
(comment
  (let [tabled
        (into {}
              (x/by-key (juxt :Pclass :Sex)
                        :Age
                        (comp (remove nil?)
                              (x/into [])))
              data)]
    (doseq [[[pclass sex] ages] tabled]
      (-> (c/histogram ages
           :title (str "Pclass = " pclass " sex = " sex)
           :x-label "Age bin"
           :y-label "Age"
           :nbins 20)
          (c/set-y-range 0 45)
          i/view)))
  )
;; Get median age per Pclass/sex to fill in nil ages:
(def age-guess (into {}
                     (x/by-key (juxt :Pclass :Sex)
                               :Age
                               (comp (remove nil?)
                                     (x/into (sorted-set))
                                     (map (fn [all]
                                            (nth (seq all)
                                                 (long (/ (count all) 2)))))))
                     data))
(comment
  age-guess
  #_{[3 "male"] 29,
     [1 "female"] 38,
     [3 "female"] 24,
     [1 "male"] 39,
     [2 "female"] 29,
     [2 "male"] 34}
  )

(defn fill-in-age [row]
  (get age-guess [(:Pclass row) (:Sex row)]))

(def xfAge
  (map (fn [row]
         (update row :Age #(or % (fill-in-age row))))))

;; Fill in ages in data to be able to make age bands
(def data-ages
  (into []
        xfAge
        data))

;; 5 bin on range 0 to 80 is 16 per bin
(def age-bands {[0  16] :age-0-16
                [16 32] :age-16-32
                [32 48] :age-32-48
                [48 64] :age-48-64
                [64 80] :age-64-80})

(def xfAgeBand
  (map (fn [row]
         (assoc row :AgeBand
                (some
                 (fn [[[low high] band]]
                   (when (and (< low (:Age row))
                              (<= (:Age row) high))
                     band))
                 age-bands)))))

;; Correlation AgeBand and Survived
(comment
  (->> data
       (into []
             (comp xfAge
                   xfAgeBand
                   (x/by-key :AgeBand
                             :Survived
                             (comp x/avg
                                   (map double)))))
       (sort-by second >))
  #_([:age-0-16 0.55]
     [:age-48-64 0.4347826086956522]
     [:age-32-48 0.4052863436123348]
     [:age-16-32 0.3388429752066116]
     [:age-64-80 0.09090909090909091])
  )

;; Create IsAlone feature
(def xfIsAlone
  (map (fn [row]
         (assoc row :IsAlone (if (zero? (+ (:SibSp row)
                                           (:Parch row)))
                               :alone
                               :not-alone)))))

;; Relation IsAlone and Survived
(comment
  (into {}
        (comp xfIsAlone
              (x/by-key :IsAlone
                        :Survived
                        (comp x/avg
                              (map double))))
        data)
  ;;{:not-alone 0.5056497175141244, :alone 0.3035381750465549}
  )

;; Fill in missing embarked values by choosing the most common
(comment
  (->> data
       (map :Embarked)
       frequencies
       (sort-by val >)
       first)
  ;; ["S" 644]
  )

(def xfEmbarked
  (map (fn [row]
         (update row :Embarked #(keyword (or % "S"))))))

;; FareBands
(comment
  (let [parts (stats/quantile (map :Fare data-ages) :probs [0 0.25 0.5 0.75 1])
        bands (zipmap (map vec (partition 2 1 parts))
                      [:fare-band-0
                       :fare-band-1
                       :fare-band-2
                       :fare-band-3])]
    bands))

(def fare-bands {[0.0 7.9104] :fare-band-0,
                 [7.9104 14.4542] :fare-band-1,
                 [14.4542 31.0] :fare-band-2,
                 [31.0 512.3292] :fare-band-3})

(def xfFareBand
  (map (fn [row]
         (assoc row :FareBand
                (if (:Fare row)
                  (some
                   (fn [[[low high] band]]
                     (when (<= low (:Fare row) high)
                       band))
                   fare-bands)
                  ;; default when no Fare for passenger
                  :fare-band-1)))))

;; Transducer to clean the train data and the test data
(def xfData
  (comp xfAge
        xfAgeBand
        xfIsAlone
        xfEmbarked
        xfFareBand
        (map (fn [row]
               ;; clj-ml need keyword for category features
               (-> row
                   (update :Pclass {1 :pc-one
                                    2 :pc-two
                                    3 :pc-three})
                   (update :Survived {0 :not-survived
                                      1 :survived})
                   (update :Sex keyword))))
        (map (fn [row]
               (dissoc row :Ticket :Cabin :Name :Age :SibSp :Parch :Fare)))))

;; Make training data proper shape for ml
(def train-data
  (into []
        xfData
        data))

;; Make test-data in proper shape for ml
(def test-data
  (into []
        xfData
        (:rows test-idata)))

;; Note PassengerId is not used, it is not a feature
(def clj-ml-features
  [{:Sex [:male :female]}
   {:Age [:age-0-16
          :age-16-32
          :age-32-48
          :age-48-64
          :age-64-80]}
   {:Pclass [:pc-one :pc-two :pc-three]}
   {:Embarked [:S :C :Q]}
   {:IsAlone [:alone :not-alone]}
   {:FareBand [:fare-band-0
               :fare-band-1
               :fare-band-2
               :fare-band-3]}
   {:Survived [:survived
               :not-survived]}])

(def to-clj-ml-vector
  (apply juxt (map ffirst clj-ml-features)))

;; ds is a dataset in the shape that clj-ml/Weka requires
(def ds (-> (cm-data/make-dataset "titanic"
                                  clj-ml-features
                                  (map to-clj-ml-vector
                                       train-data))
            ;; let clj-ml know which column we want to predict
            (cm-data/dataset-set-class :Survived)))

(comment
  (cm-data/attributes ds)
  ;; (#object[weka.core.Attribute 0x10418019 "@attribute Sex {male,female}"]
  ;;  #object[weka.core.Attribute 0x76851d43 "@attribute Age {age-0-16,age-16-32,age-32-48,age-48-64,age-64-80}"]
  ;;  #object[weka.core.Attribute 0x42f64cc9 "@attribute Pclass {pc-one,pc-two,pc-three}"]
  ;;  #object[weka.core.Attribute 0x1fe50f48 "@attribute Embarked {S,C,Q}"]
  ;;  #object[weka.core.Attribute 0x3b815e5 "@attribute IsAlone {not-alone,alone}"]
  ;;  #object[weka.core.Attribute 0x5d896fd4 "@attribute FareBand {fare-band-0,fare-band-1,fare-band-2,fare-band-3}"]
  ;;  #object[weka.core.Attribute 0x2de9e0a8 "@attribute Survived {survived,not-survived}"])
  )

(def logres (cm-classifiers/make-classifier :regression :logistic))

(comment
  (def trained (cm-classifiers/classifier-train logres ds))
  (println trained)
  (.coefficients trained)

  (def ct (cm-classifiers/classifier-evaluate logres :cross-validation ds 10))
  (pprint/pprint
   ct)
  ;; :percentage-correct 79.12457912457913,

  (print
   (:confusion-matrix ct))
 ;;  === Confusion Matrix ===

 ;;   a   b   <-- classified as
 ;; 237 105 |   a = survived
 ;;  81 468 |   b = not-survived

  (println (:summary ct))
  ;; 79%
  )

(defn predict [test-data trained-classifier out-file-path]
  (let [results (->> test-data
                     (into []
                           (x/transjuxt
                            [(comp (map :PassengerId)
                                   (x/into []))
                             (comp (map to-clj-ml-vector)
                                   (map #(->> (cm-data/make-instance ds %)
                                              (cm-classifiers/classifier-classify trained-classifier)))
                                   (x/into []))]))
                     first
                     (apply map vector))]
    (->> results
         (transduce
          (mapcat (fn [[id res]]
                    [id "," (get {:survived 1
                                  :not-survived 0} res) "\n"]))
          rf/str
          "PassengerId,Survived\n")
         (spit out-file-path))))

(comment
  (let [logres (cm-classifiers/make-classifier :regression :logistic)
        trained (cm-classifiers/classifier-train logres ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate logres :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data logres "data/logres.csv")
    )
  ;; Logistic regression:
  ;; trained:  79.349%
  ;; submitted: 0.75120
  )

(comment
  (let [svm (cm-classifiers/make-classifier :support-vector-machine :smo)
        trained (cm-classifiers/classifier-train svm ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate svm :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data svm "data/svm.csv")
    )
  ;; SVM
  ;; trained:  78.676%
  ;; submitted: 0.76555
  )

(comment
  (let [knn3 (cm-classifiers/make-classifier :lazy :ibk
                                             {:num-neighbors 3})
        trained (cm-classifiers/classifier-train knn3 ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate knn3 :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data knn3 "data/knn3.csv")
    )
  ;; KNN 3
  ;; trained:  78.451%
  ;; submitted: 0.73206
  )

(comment
  (let [bayes (cm-classifiers/make-classifier :bayes :naive)
        trained (cm-classifiers/classifier-train bayes ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate bayes :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data bayes "data/bayes.csv")
    )
  ;; Naive Bayes
  ;; trained:  78.339%
  ;; submitted: 0.70813
  )

(comment
  (let [dectree (cm-classifiers/make-classifier :decision-tree :c45)
        trained (cm-classifiers/classifier-train dectree ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate dectree :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data dectree "data/dectree.csv")
    )
  ;; Decision Tree
  ;; trained:  81.145%
  ;; submitted: 0.77990
  )

(comment
  (let [random-forest (doto (cm-classifiers/make-classifier :decision-tree :random-forest)
                        (.setNumTrees 100))
        trained (cm-classifiers/classifier-train random-forest ds)]
    (println "Trained info")
    (println trained)
    (let [evaluate (cm-classifiers/classifier-evaluate random-forest :cross-validation ds 10)]
      (pprint/pprint evaluate)
      (print
       (:confusion-matrix evaluate)))

    (predict test-data random-forest "data/randomforest.csv")
    )
  ;; Random Forest
  ;; trained:  78.563%
  ;; submitted: 0.77990
  )


;; blogpost examples:
(comment
  (-> (map :Survived data)
      frequencies)
  ;;=> {0 549, 1 342}

  (-> (map (juxt :Sex :Survived) data)
      frequencies)
  ;;=> {["male" 0] 468, ["female" 1] 233, ["female" 0] 81, ["male" 1] 109}
  (let [by-sex (group-by :Sex data)
        survived (zipmap (keys by-sex)
                         (->> by-sex
                              vals
                              (map #(-> (map :Survived %)
                                        frequencies))))]
    survived)
  ;;=> {"male" {0 468, 1 109}, "female" {1 233, 0 81}}
  (reduce
   (fn [acc {:keys [Sex Survived] :as row}]
     (update-in acc [Sex Survived] (fnil inc 0)))
   {}
   data)
  ;;=> {"male" {0 468, 1 109}, "female" {1 233, 0 81}}

  (require '[net.cgrand.xforms :as x])
  (into {}
        (x/by-key :Sex
                  (comp (x/by-key :Survived
                                  x/count)
                        (x/into {})))
        data)
  ;; => {"male" {0 468, 1 109}, "female" {1 233, 0 81}}

  (def xFrequencies (comp (x/by-key identity
                                    x/count)
                          (x/into {})))
  (into {}
        (x/by-key :Sex
                  :Survived
                  (x/transjuxt {:chance (comp x/avg
                                              (map double))
                                :counts xFrequencies}))
        data)
  ;; {"male" {:chance 0.1889081455805893, :counts {0 468, 1 109}},
  ;;  "female" {:chance 0.7420382165605096, :counts {1 233, 0 81}}}


  (def ageStats
    (comp (map :Age)
          (filter identity) ;; some Age values are missing
          (x/transjuxt {:mean (comp x/avg
                                    (map double))
                        :std-dev x/sd})))
  (into {}
        ageStats
        data)
  ;;=> {:mean 29.69911764705882, :std-dev 14.526497332334035}

  (into {}
        (x/by-key :Sex
                  ageStats)
        data)
  ;;=> {"male" {:mean 30.72664459161148, :std-dev 14.678200823816606},
  ;;    "female" {:mean 27.915708812260537, :std-dev 14.110146457544133}}
  )
