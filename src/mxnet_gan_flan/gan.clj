;;
;; Licensed to the Apache Software Foundation (ASF) under one or more
;; contributor license agreements.  See the NOTICE file distributed with
;; this work for additional information regarding copyright ownership.
;; The ASF licenses this file to You under the Apache License, Version 2.0
;; (the "License"); you may not use this file except in compliance with
;; the License.  You may obtain a copy of the License at
;;
;;    http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.
;;

(ns mxnet-gan-flan.gan
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.optimizer :as opt]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.util :as util]
            [mxnet-gan-flan.viz :as viz]
            [org.apache.clojure-mxnet.context :as context]
            [think.image.pixel :as pixel]
            [mikera.image.core :as img])
  (:gen-class))

;; based off of https://medium.com/@julsimon/generative-adversarial-networks-on-apache-mxnet-part-1-b6d39e6b5df1

;;; This will load 128x128 images and generator for them
;;; You might want to start off small then that first - with 28x28 images first. In that case, use the network layers
;; from here https://github.com/apache/incubator-mxnet/blob/master/contrib/clojure-package/examples/gan/src/gan/gan_mnist.clj#L90
;; and increase your batch size to 10 or 100. You will also need a rec file that has the same dimensions


(def data-dir "data/")
(def output-path "results/")
(def model-path "model/")
(def batch-size 5)
(def num-epoch 10)

(io/make-parents (str output-path "gout"))
(io/make-parents (str model-path "test"))

(defn last-saved-model-number []
  (some->> "model/"
           clojure.java.io/file
           file-seq
           (filter #(.isFile %))
           (map #(.getName %))
           (filter #(clojure.string/includes? %  "model-d"))
           (map #(re-seq #"\d{4}" %))
           (map first)
           (map #(when % (Integer/parseInt %)))
           (sort)
           (last)))

(def flan-iter (mx-io/image-record-iter {:path-imgrec "flan-128.rec"
                                         :data-shape [3 128 128]
                                         :batch-size batch-size
                                         :shuffle true}))

(defn normalize-rgb [x]
  (/ (- x 128.0) 128.0))

(defn normalize-rgb-ndarray [nda]
  (let [nda-shape (ndarray/shape-vec nda)
        new-values (mapv #(normalize-rgb %) (ndarray/->vec nda))]
    (ndarray/array new-values nda-shape)))


(defn denormalize-rgb [x]
  (+ (* x 128.0) 128.0))

(defn clip [x]
  (cond
    (< x 0) 0
    (> x 255) 255
    :else (int x)))


(defn postprocess-image [img]
  (let [datas (ndarray/->vec img)
        image-shape (mx-shape/->vec (ndarray/shape img))
        spatial-size (* (get image-shape 2) (get image-shape 3))
        pics (doall (partition (* 3 spatial-size) datas))
        pixels  (mapv
                 (fn [pic]
                   (let [[rs gs bs] (doall (partition spatial-size pic))
                         this-pixels (mapv (fn [r g b]
                                              (pixel/pack-pixel
                                               (int (clip (denormalize-rgb r)))
                                               (int (clip (denormalize-rgb g)))
                                               (int (clip (denormalize-rgb b)))
                                               (int 255)))
                                           rs gs bs)]
                     this-pixels))
                 pics)
        new-pixels (into [] (flatten pixels))
        new-image (img/new-image (* 1 (get image-shape 3)) (* batch-size (get image-shape 2)))
        _  (img/set-pixels new-image (int-array new-pixels))]
    new-image))

(defn postprocess-write-img [img filename]
  (img/write (-> (postprocess-image img)
                 (img/zoom 1.5)) filename "png"))


(def rand-noise-iter (mx-io/rand-iter [batch-size 100 1 1]))

(comment

  ;;This is for figuring out the convolution and deconvolution layers to convert the image sizes

  (defn conv-output-size [input-size kernel-size padding stride]
    (float (inc (/ (- (+ input-size (* 2 padding)) kernel-size) stride))))

  ;; Calcing the layer sizes for discriminator
  (conv-output-size 28 4 3 2) ;=> 16
  (conv-output-size 16 4 1 2) ;=> 8
  (conv-output-size 8 4 1 2) ;=> 4.0
  (conv-output-size 4 4 0 1) ;=> 1

  ;;;; for 128
  (conv-output-size 128 4 3 2) ;=> 66
  (conv-output-size 66 4 2 2) ;=> 34.0
  (conv-output-size 34 4 0 2) ;=> 16
  (conv-output-size 16 4 1 2) ;=> 8
  (conv-output-size 8 4 1 2) ;=> 4.0
  (conv-output-size 4 4 0 1) ;=> 1

  ;; Calcing the layer sizes for generator
  (defn deconv-output-size [input-size kernel-size padding stride]
    (-
     (+ (* stride (- input-size 1))
        kernel-size)
     (* 2 padding)))

  (deconv-output-size 1 4 0 1) ;=> 4
  (deconv-output-size 4 4 1 2) ;=> 8
  (deconv-output-size 8 4 1 2) ;=> 16
  (deconv-output-size 16 4 3 2)) ;=> 28


(def ndf 28) ;; image height /width
(def nc 3) ;; number of channels
(def eps (float (+ 1e-5  1e-12)))
(def lr  0.0005) ;; learning rate
(def beta1 0.5)

(def label (sym/variable "label"))

(defn discriminator []
  (as-> (sym/variable "data") data
    (sym/convolution "d2" {:data data :kernel [4 4] :pad [3 3] :stride [2 2] :num-filter (* 2 ndf) :no-bias true})
    (sym/batch-norm "dbn2" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact1" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d3" {:data data :kernel [4 4] :pad [2 2] :stride [2 2] :num-filter (* 3 ndf) :no-bias true})
    (sym/batch-norm "dbn3" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact3" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d4" {:data data :kernel [4 4] :pad [0 0] :stride [2 2] :num-filter (* 3 ndf) :no-bias true})
    (sym/batch-norm "dbn4" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact4" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d5" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 3 ndf) :no-bias true})
    (sym/batch-norm "dbn5" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact5" {:data data :act_type "leaky" :slope 0.2})

        (sym/convolution "d6" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 3 ndf) :no-bias true})
    (sym/batch-norm "dbn6" {:data data :fix-gamma true :eps eps})
    (sym/leaky-re-lu "dact6" {:data data :act_type "leaky" :slope 0.2})

    (sym/convolution "d7" {:data data :kernel [4 4] :pad [0 0] :stride [1 1] :num-filter (* 4 ndf) :no-bias true})
    (sym/flatten "flt" {:data data})

    (sym/fully-connected "fc" {:data data :num-hidden 1 :no-bias false})
    (sym/logistic-regression-output "dloss" {:data data :label label})))

(defn generator []
  (as-> (sym/variable "rand") data
    (sym/deconvolution "g1" {:data data :kernel [4 4]  :pad [0 0] :stride [1 1] :num-filter (* 4 ndf) :no-bias true})
    (sym/batch-norm "gbn1" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact1" {:data data :act-type "relu"})

    (sym/deconvolution "g2" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter (* 2 ndf) :no-bias true})
    (sym/batch-norm "gbn2" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact2" {:data data :act-type "relu"})

    (sym/deconvolution "g3" {:data data :kernel [4 4] :pad [1 1] :stride [2 2] :num-filter ndf :no-bias true})
    (sym/batch-norm "gbn3" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact3" {:data data :act-type "relu"})

    (sym/deconvolution "g4" {:data data :kernel [4 4] :pad [0 0] :stride [2 2] :num-filter ndf :no-bias true})
    (sym/batch-norm "gbn4" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact4" {:data data :act-type "relu"})

    (sym/deconvolution "g5" {:data data :kernel [4 4] :pad [2 2] :stride [2 2] :num-filter ndf :no-bias true})
    (sym/batch-norm "gbn5" {:data data :fix-gamma true :eps eps})
    (sym/activation "gact5" {:data data :act-type "relu"})

    (sym/deconvolution "g7" {:data data :kernel [4 4] :pad [3 3] :stride [2 2] :num-filter nc :no-bias true})
    (sym/activation "gact7" {:data data :act-type "tanh"})))

(defn save-img-gout [i n x]
  (do
    (viz/im-sav {:title (str "gout-" i "-" n)
                 :output-path output-path
                 :x x
                 :flip false})))

(defn save-img-diff [i n x]
  (do (viz/im-sav {:title (str "diff-" i "-" n)
                   :output-path output-path
                   :x x
                   :flip false})))

(defn save-img-data [i n batch]
  (do (viz/im-sav {:title (str "data-" i "-" n)
                   :output-path output-path
                   :x batch
                   :flip false})))

(defn calc-diff [i n diff-d]
  (let [diff (ndarray/copy diff-d)
        arr (ndarray/->vec diff)
        mean (/ (apply + arr) (count arr))
        std (let [tmp-a (map #(* (- % mean) (- % mean)) arr)]
              (float (Math/sqrt (/ (apply + tmp-a) (count tmp-a)))))]
    (let [calc-diff (ndarray/+ (ndarray/div (ndarray/- diff mean) std) 0.5)]

      (save-img-diff i n calc-diff))))

(defn train [devs]
  (let [last-train-num (last-saved-model-number)
        _ (println "The last saved trained epoch is " last-train-num)
        mod-d  (-> (if last-train-num
                     (do
                       (println "Loading discriminator from checkpoint of epoch " last-train-num)
                       (m/load-checkpoint {:contexts devs
                                           :data-names ["data"]
                                           :label-names ["label"]
                                           :prefix (str model-path "model-d")
                                           :epoch last-train-num
                                           :load-optimizer-states true}))
                      (m/module (discriminator) {:contexts devs :data-names ["data"] :label-names ["label"]}))
                   (m/bind {:data-shapes (mx-io/provide-data flan-iter)
                            :label-shapes (mx-io/provide-label flan-iter)
                            :inputs-need-grad true})
                   (m/init-params {:initializer (init/normal 0.02)})
                   (m/init-optimizer {:optimizer (opt/adam {:learning-rate lr :wd 0.0 :beta1 beta1})}))
        mod-g (-> (if last-train-num
                    (do
                     (println "Loading generator from checkpoint of epoch " last-train-num)
                     (m/load-checkpoint {:contexts devs
                                         :data-names ["rand"]
                                         :label-names [""]
                                         :prefix (str model-path "model-g")
                                         :epoch last-train-num
                                         :load-optimizer-states true}))
                    (m/module (generator) {:contexts devs :data-names ["rand"] :label-names nil}))
                  (m/bind {:data-shapes (mx-io/provide-data rand-noise-iter)})
                  (m/init-params {:initializer (init/normal 0.02)})
                  (m/init-optimizer {:optimizer (opt/adam {:learning-rate lr :wd 0.0 :beta1 beta1})}))]

    (println "Training for " num-epoch " epochs...")
    (doseq [i (if last-train-num
                (range (inc last-train-num) (inc (+ last-train-num num-epoch)))
                (range num-epoch))]
      (mx-io/reduce-batches flan-iter
                            (fn [n batch]
                              (let [rbatch (mx-io/next rand-noise-iter)
                                    dbatch (mapv normalize-rgb-ndarray (mx-io/batch-data batch))
                                    out-g (-> mod-g
                                              (m/forward rbatch)
                                              (m/outputs))
                                    ;; update the discriminiator on the fake
                                    grads-f  (mapv #(ndarray/copy (first %)) (-> mod-d
                                                                                 (m/forward {:data (first out-g) :label [(ndarray/zeros [batch-size])]})
                                                                                 (m/backward)
                                                                                 (m/grad-arrays)))
                                   ;; update the discrimintator on the real
                                    grads-r (-> mod-d
                                                (m/forward {:data dbatch :label [(ndarray/ones [batch-size])]})
                                                (m/backward)
                                                (m/grad-arrays))
                                    _ (mapv (fn [real fake] (let [r (first real)]
                                                              (ndarray/set r (ndarray/+ r fake)))) grads-r grads-f)
                                    _ (m/update mod-d)
                                   ;; update the generator
                                    diff-d (-> mod-d
                                               (m/forward {:data (first out-g) :label [(ndarray/ones [batch-size])]})
                                               (m/backward)
                                               (m/input-grads))
                                    _ (-> mod-g
                                          (m/backward (first diff-d))
                                          (m/update))]
                                (when (zero? n)
                                  (println "iteration = " i  "number = " n)
                                  (save-img-gout i n (ndarray/copy (ffirst out-g)))
                                  (save-img-data i n (first dbatch))
                                  (calc-diff i n (ffirst diff-d))
                                  (m/save-checkpoint mod-g {:prefix (str model-path "model-g") :epoch i :save-opt-states true})
                                  (m/save-checkpoint mod-d {:prefix (str model-path "model-d") :epoch i :save-opt-states true}))
                                (inc n)))))))

(defn -main [& args]
  (let [[dev dev-num] args
        devs (if (= dev ":gpu")
               (mapv #(context/gpu %) (range (Integer/parseInt (or dev-num "1"))))
               (mapv #(context/cpu %) (range (Integer/parseInt (or dev-num "1")))))]
    (println "Running with context devices of" devs)
    (train devs)))


(defn explore
  "Use this to explore your models that you have trained.
   Use the epoch that you wish to load and the number of pictures that you want to generate."
  ([epoch num]
   (explore (str model-path "model-g") epoch num))
  ([prefix epoch num]

   (let [mod-g (-> (m/load-checkpoint {:contexts [(context/default-context)]
                                       :data-names ["rand"]
                                       :label-names [""]
                                       :prefix prefix
                                       :epoch epoch})
                   (m/bind {:data-shapes (mx-io/provide-data rand-noise-iter)})
                   (m/init-params {:initializer (init/normal 0.02)})
                   (m/init-optimizer {:optimizer (opt/adam {:learning-rate lr :wd 0.0 :beta1 beta1})}))]

     (println "Generating images from " epoch)
     (dotimes [i num]
       (let [rbatch (mx-io/next rand-noise-iter)
             out-g (-> mod-g
                       (m/forward rbatch)
                       (m/outputs))]
         (viz/im-sav {:title (str "explore-" epoch "-" i)
                      :output-path output-path
                      :x (ffirst out-g)
                      :flip false})
         (-> (img/load-image (str "results/explore-" epoch "-" i ".jpg"))
             (img/show)))))))

(defn explore-pretrained
  "Use this to explore the pretrained model of flans.
   Specify the number of pictures that you want to generate."
  [num]
  (explore (str "pre-trained/" "model-g") 195 num))

(comment
  (train [(context/cpu)])


  (explore 4 1)
  (explore-pretrained 1)

  )
