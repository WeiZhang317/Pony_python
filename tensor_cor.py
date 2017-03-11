import tensorflow as tf


def tf_var_graph(x, n):
    x = x - tf.reduce_mean(x)
    xx_sum = tf.reduce_sum(x * x)
    x_sum = tf.reduce_sum(x)

    var = (xx_sum - (x_sum * x_sum) / n) / (n - 1)
    return var


def tf_var(data,sess=None):
    # todo check parameter
    x = tf.placeholder(tf.float32)
    n = tf.placeholder(tf.float32)
    f = tf_var_graph(x, n)
    if not sess : return (f, {x: data, n: len(data)})
    return sess.run(f, {x: data, n: len(data)})

def tf_cov_graph(x, y, n):
    x = x - tf.reduce_mean(x)
    y = y - tf.reduce_mean(y)

    xy_sum = tf.reduce_sum(x * y)
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)

    return (xy_sum - (x_sum * y_sum) / n) / (n - 1)


def tf_cov(dataX, dataY,sess=None):
    # todo check parameter
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    n = tf.placeholder(tf.float32)
    f = tf_cov_graph(x, y, n)

    if not sess : return (f, {x: dataX, y: dataY, n: len(dataX)})
    return sess.run(f, {x: dataX, y: dataY, n: len(dataX)})


def tf_cor_graph(x, y, n):
    # pearson
    x = x - tf.reduce_mean(x)
    y = y - tf.reduce_mean(y)

    xy_sum = tf.reduce_sum(x * y)
    x_sum = tf.reduce_sum(x)
    y_sum = tf.reduce_sum(y)

    cov = (xy_sum - (x_sum * y_sum) / n) / (n - 1)

    xx_sum = tf.reduce_sum(x * x)
    x_var = (xx_sum - (x_sum * x_sum) / n) / (n - 1)
    x_sd = tf.sqrt(x_var)

    yy_sum = tf.reduce_sum(y * y)
    y_var = (yy_sum - (y_sum * y_sum) / n) / (n - 1)
    y_sd = tf.sqrt(y_var)

    return cov / (x_sd * y_sd) # pearson

def tf_cor(dataX, dataY,sess=None):
    # pearson
    # todo check parameter
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    n = tf.placeholder(tf.float32)
    f = tf_cor_graph(x, y, n)
    if not sess : return (f, {x: dataX, y: dataY, n: len(dataX)})
    return sess.run(f, {x: dataX, y: dataY, n: len(dataX)})

def tf_cor_spearman(dataX, dataY):
    # The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables;
    pass


def tf_cor_kendall(dataX, dataY):
    pass

if __name__ == "__main__":
    sess = tf.Session()
    f,feed = tf_var([1, 2, 3, 4, 5])
    print sess.run(f,feed)
    f,feed =  tf_cov([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    print sess.run(f,feed)
    f,feed = tf_cor([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    print sess.run(f,feed)
    print

    print tf_var([1, 2, 3, 4, 5],sess)
    print tf_cov([1, 2, 3, 4, 5], [1, 2, 3, 4, 5],sess)
    print tf_cor([1, 2, 3, 4, 5], [1, 2, 3, 4, 5],sess)
    print
    print tf_var([1, 3, 9, 8, 0],sess)
    print tf_cov([1, 3, 9, 8, 0], [1, 3, 9, 8, 0],sess)
    print tf_cor([1, 3, 9, 8, 0], [1, 3, 9, 8, 0],sess)
    print
    print tf_var([1, 3, 9, 8, 0],sess)
    print tf_cov([1, 3, 9, 8, 0], [2, 3, 8, 7, 1],sess)
    print tf_cor([1, 3, 9, 8, 0], [2, 3, 8, 7, 1],sess)
    print
    print tf_var([1, 3, 9, 8, 0],sess)
    print tf_cov([1, 3, 9, 8, 0], [2, 1, 3, 5, 9],sess)
    print tf_cor([1, 3, 9, 8, 0], [2, 1, 3, 5, 9],sess)






