(set-logic QF_LIA)
(declare-const x Int) (declare-const y Int) (assert ( or ( < ( * 0 y ) 1 ) ( = ( + ( * 0 x ) ( * 0 y ) ) 1 ) ) )
(check-sat)
(exit)
