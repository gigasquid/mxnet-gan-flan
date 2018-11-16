(ns gan.layers
  (:require [org.apache.clojure-mxnet.symbol :as sym]))

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

(defn discriminator-128 []
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

(defn generator-128 []
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
