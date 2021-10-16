(set-logic LIA)

(declare-const z Int)
(declare-const z! Int)
(declare-const y Int)
(declare-const y! Int)
(declare-const x Int)
(declare-const x! Int)
(declare-const junk_0 Int)
(declare-const junk_0! Int)

(declare-const z_0 Int)
(declare-const y_29 Int)
(declare-const y_21 Int)
(declare-const y_0 Int)
(declare-const x_28 Int)
(declare-const x_19 Int)
(declare-const x_18 Int)
(declare-const junk_0_27 Int)
(declare-const junk_0_22 Int)
(declare-const junk_0_20 Int)
(declare-const junk_0_17 Int)

(define-fun inv-f((junk_0 Int)(x Int)(y Int)(z Int)) Bool
SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop

)

(define-fun pre-f ((junk_0 Int)(x Int)(y Int)(z Int)(junk_0_17 Int)(junk_0_20 Int)(junk_0_22 Int)(junk_0_27 Int)(x_18 Int)(x_19 Int)(x_28 Int)(y_0 Int)(y_21 Int)(y_29 Int)(z_0 Int)) Bool
  (and
    (= x x_18)
    (= junk_0 junk_0_17)
    (= junk_0_17 2)
    (= x_18 0)
  )
)

(define-fun trans-f ((junk_0 Int)(x Int)(y Int)(z Int)(junk_0! Int)(x! Int)(y! Int)(z! Int)(junk_0_17 Int)(junk_0_20 Int)(junk_0_22 Int)(junk_0_27 Int)(x_18 Int)(x_19 Int)(x_28 Int)(y_0 Int)(y_21 Int)(y_29 Int)(z_0 Int)) Bool
  (or
    (and
      (= junk_0_27 junk_0)
      (= x_28 x)
      (= y_29 y)
      (= y_29 y!)
      (= x_28 x!)
      (= junk_0_27 junk_0!)
      (= z z!)
      (= y y!)
      (= junk_0 junk_0!)
    )
    (and
      (= junk_0_27 junk_0)
      (= x_28 x)
      (= y_29 y)
      (< x_28 500)
      (= x_19 (+ x_28 1))
      (= junk_0_20 (+ 403 junk_0_27))
      (not (<= z_0 y_29))
      (= y_29 y!)
      (= x_19 x!)
      (= junk_0_20 junk_0!)
      (= z z_0)
      (= z! z_0)
    )
    (and
      (= junk_0_27 junk_0)
      (= x_28 x)
      (= y_29 y)
      (< x_28 500)
      (= x_19 (+ x_28 1))
      (= junk_0_20 (+ 403 junk_0_27))
      (<= z_0 y_29)
      (= y_21 z_0)
      (= junk_0_22 (+ junk_0_20 junk_0_20))
      (= y_21 y!)
      (= x_19 x!)
      (= junk_0_22 junk_0!)
      (= z z_0)
      (= z! z_0)
    )
  )
)

(define-fun post-f ((junk_0 Int)(x Int)(y Int)(z Int)(junk_0_17 Int)(junk_0_20 Int)(junk_0_22 Int)(junk_0_27 Int)(x_18 Int)(x_19 Int)(x_28 Int)(y_0 Int)(y_21 Int)(y_29 Int)(z_0 Int)) Bool
  (or
    (not
      (and
        (= x x_28)
        (= y y_29)
        (= z z_0)
      )
    )
    (not
      (and
        (not (< x_28 500))
        (not (>= z_0 y_29))
      )
    )
  )
)



SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
(assert  (not
  (=>
    (pre-f junk_0 x y z junk_0_17 junk_0_20 junk_0_22 junk_0_27 x_18 x_19 x_28 y_0 y_21 y_29 z_0 )
    (inv-f junk_0 x y z )
  )
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
(assert  (not
  (=>
    (and
      (inv-f junk_0 x y z )
      (trans-f junk_0 x y z junk_0! x! y! z! junk_0_17 junk_0_20 junk_0_22 junk_0_27 x_18 x_19 x_28 y_0 y_21 y_29 z_0 )
    )
    (inv-f junk_0! x! y! z! )
  )
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
(assert  (not
  (=>
    (inv-f junk_0 x y z )
    (post-f junk_0 x y z junk_0_17 junk_0_20 junk_0_22 junk_0_27 x_18 x_19 x_28 y_0 y_21 y_29 z_0 )
  )
))

