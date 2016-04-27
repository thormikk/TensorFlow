import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.Variable(0,name="a")
    b = tf.Variable(1,name="b")
    c = tf.Variable(1,name="c")
    sum = tf.add(a,b)
    event1 = tf.assign(c,sum)
    event2 = tf.assign(a,b)
    event3 = tf.assign(b,c)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        for _ in range(10):
            sess.run(event1)
            sess.run(event2)
            sess.run(event3)
        print(a.eval())
        writer = tf.train.SummaryWriter("",sess.graph)
    sess.close()


f = tf.Graph()
with f.as_default():
    a1 = tf.constant(0,name="a1")
    b1 = tf.constant(1,name="b1")
    with tf.Session() as sess1:
        for _ in range(10):
            c1 = tf.add(a1,b1,name="c1")
            a1 = b1
            b1 = c1.eval()
        print(a1)
        writer = tf.train.SummaryWriter("",sess1.graph)

h = tf.Graph()
with h.as_default():
    a2 = tf.constant(0,name="a2")
    b2 = tf.constant(1,name="b2")
    for _ in range(10):
        c2 = tf.add(a2,b2)
        a2 = b2
        b2 = c2
    with tf.Session() as sess2:
        print(a2.eval())
        writer = tf.train.SummaryWriter("",sess2.graph)













