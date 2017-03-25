import tensorflow as tf


def tf_var(x):
    """Take `x` as input tensor, produce a tensor used to calculate Variance.
    
    For example:
    ```python
    x = tf.placeholder(tf.float32)
    f = tf_var(x)
    ```

    :param x: Input tensor 
    :return: A `Tensor` that used to calculate Variance.
    """
    x = x - tf.reduce_mean(x)
    n = tf.to_float(tf.size(x))
    var = tf.reduce_sum(x * x) / (n - 1)
    return var


def tf_var_run(data, sess=None):
    """Calculate Variance of `data`.
    Return Variance if sess is provided, return tensor need for calculating the Variance.
    
    For example:
    ```python
    with tf.Session() as sess:
        print tf_var_run([1, 2, 3, 4, 5],sess)
        print sess.run(tf_var_run([1, 2, 3, 4, 5]))
    ```

    :param data: number array
    :param sess: TensorFlow session
    :return: Variance if sess is provided, a tensor used for calculating the Variance otherwise.
    """
    # todo check parameter
    x = tf.placeholder(tf.float32)
    f = tf_var(x)
    if not sess: return (f, {x: data})
    return sess.run(f, {x: data})


def tf_cov(x, y):
    """Take variables `x`,`y` as input tensor, produce the tensor used to calculate Covariance.
    
    For example:
    ```python
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    f = tf_cov(x, y)
    
    :param x: input tensor
    :param y: input tensor
    :return: A `Tensor` that used to calculate Covariance.
    """
    n = tf.to_float(tf.size(x))
    x = x - tf.reduce_mean(x)
    y = y - tf.reduce_mean(y)
    cov = tf.reduce_sum(x * y) / (n - 1)
    return cov


def tf_cov_run(dataX, dataY, sess=None):
    """Calculate Covariance between `dataX` and `dataY`.
    Return Covariance if sess is provided, return a tensor used for calculating the Covariance between `dataX` and `dataY`.
    
    For example:
    ```python
    with tf.Session() as sess:
        print tf_cov_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], sess)
        print sess.run(tf_cov_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))
    ```
    :param dataX: number array
    :param dataY: number array
    :param sess: TensorFlow Session
    :return: Covariance if sess is provided, a tensor used for calculating the Covariance otherwise.
    """
    # todo check parameter
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    f = tf_cov(x, y)
    if not sess: return (f, {x: dataX, y: dataY})
    return sess.run(f, {x: dataX, y: dataY})


def tf_cor(x, y):
    # Pearson product-moment correlation coefficient between two variables.

    """Take variables `x` and `y` as input tensor,
    produce a tensor used to calculate Pearson correlation coefficient between two variables `x` and `y`.
    
    For example:
    ```python
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    f = tf_cov(x, y)
    ```
    
    :param x: input tensor
    :param y: input tensor
    :return: A `Tensor` that used to calculate Pearson correlation coefficient between two variables `x` and `y`.
    """
    n = tf.to_float(tf.size(x))
    x = x - tf.reduce_mean(x)
    y = y - tf.reduce_mean(y)

    cov = tf.reduce_sum(x * y) / (n - 1)
    x_var = tf.reduce_sum(x * x) / (n - 1)
    x_sd = tf.sqrt(x_var)

    y_var = tf.reduce_sum(y * y) / (n - 1)
    y_sd = tf.sqrt(y_var)

    cor = cov / (x_sd * y_sd)
    return cor


def tf_cor_run(dataX, dataY, sess=None):
    # Pearson product-moment correlation coefficient between two variables.

    """Calculate Pearson correlation coefficient between `dataX` and `dataY`.
    Return Pearson correlation coefficient if sess is provided, 
    return a tensor used for calculating the Pearson correlation coefficient between `dataX` and `dataY`.
    
    For example:
    ```python
    with tf.Session() as sess:
        print tf_cor_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], sess)
        print sess.run(tf_cor_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))
    ```
    
    :param dataX: number array
    :param dataY: number array
    :param sess: TensorFlow Session
    :return: Pearson correlation coefficient if sess is provided, 
    a tensor used for calculating the Pearson correlation coefficient between `dataX` and `dataY` otherwise.
    """
    # todo check parameter
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    f = tf_cor(x, y)
    if not sess: return (f, {x: dataX, y: dataY})
    return sess.run(f, {x: dataX, y: dataY})


def tf_cor_spearman(dataX, dataY):
    # The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables;
    pass


def tf_cor_kendall(dataX, dataY, sess=None):
    pass


def tf_cor_matrix(x):
    # Pearson product-moment correlation coefficient between two columns in one matrix.

    """ Take matrix `x` as input tensor,
    produce a tensor used to calculate Pearson correlation coefficient between two columns in one matrix `x`.
    
    For example:
    ```python
    x = tf.placeholder(tf.float32)
    f = tf_cor_matrix(x)
    ```
    :param x: input tensor
    :return: A `Tensor` that used to calculate Pearson correlation coefficient between two columns in matrix `x`.
    """

    # todo check dim
    n = tf.to_float(tf.shape(x)[1])
    x_mean = tf.reduce_mean(x, 1)
    x = x - tf.expand_dims(x_mean, 1)

    x_var = tf.reduce_sum(x * x, 1) / (n - 1)
    x_sd = tf.sqrt(x_var)

    def xix_cor(xi):
        xix_cov = tf.reduce_sum(xi * x, 1) / (n - 1)

        xi_var = tf.reduce_sum(xi * xi) / (n - 1)
        xi_sd = tf.sqrt(xi_var)

        xix_cor = xix_cov / (xi_sd * x_sd)
        return xix_cor

    result = tf.map_fn(xix_cor, x)
    return result


def tf_cor_matrix_run(data, sess=None):
    # Pearson product-moment correlation coefficient between two columns in a matrix.

    """Calculate Pearson correlation coefficient between two columns in a matrix `data`.
    Return Pearson correlation coefficient if sess is provided, 
    return a tensor used for calculating the Pearson correlation coefficient between two columns in a matrix `data`.
    
    For example:
    ```python
    with tf.Session() as sess:
        print tf_cor_matrix_run([[1, 3, 9], [8, 0, 1],[2, 1, 3], [5, 9, 0]], sess)
        print sess.run(tf_cor_matrix_run([[1, 3, 9], [8, 0, 1],[2, 1, 3], [5, 9, 0]]))
    ```
    :param data: a matrix
    :param sess: TensorFlow Session
    :return: Pearson correlation coefficient if sess is provided, 
    a tensor used for calculating the Pearson correlation coefficient between two columns in matrix otherwise.
    """

    # todo check parameter
    x = tf.placeholder(tf.float32)
    f = tf_cor_matrix(x)
    if not sess: return (f, {x: data})
    return sess.run(f, {x: data})


if __name__ == "__main__":

    with tf.Session() as sess:

        print tf_var_run([1, 2, 3, 4, 5], sess)
        print tf_cov_run([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], sess)
        print tf_cor_run([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], sess)
        print tf_cov_run([1, 3, 9, 8, 0], [2, 1, 3, 5, 9], sess)
        print tf_cor_run([1, 3, 9, 8, 0], [2, 1, 3, 5, 9], sess)
        print

        data = [[1, 3, 9], [8, 0, 1], [2, 1, 3], [5, 9, 0]]

        for i in range(4):
            ret = []
            for j in range(4):
                ret.append(tf_cor_run(data[i], data[j], sess))
            print  ret

        print
        print tf_cor_matrix_run(data, sess)
        # todo use compensated summation(Kahan summation algorithm) to get more accuracy

    print "var sample"
    with tf.Session() as sess:

        print tf_var_run([1, 2, 3, 4, 5], sess)
        f, data = tf_var_run([1, 2, 3, 4, 5])
        print sess.run(f, data)

    print "cov sample"
    with tf.Session() as sess:

        print tf_cov_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], sess)
        f, data = tf_cov_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        print sess.run(f, data)

    print "cor sample"
    with tf.Session() as sess:

        print tf_cor_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], sess)
        f, data = tf_cor_run([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        print sess.run(tf_cor_run(f, data))

    print "cor_matrix sample"
    with tf.Session() as sess:

        print tf_cor_matrix_run([[1, 3, 9], [8, 0, 1],[2, 1, 3], [5, 9, 0]], sess)
        f, data = tf_cor_matrix_run([[1, 3, 9], [8, 0, 1],[2, 1, 3], [5, 9, 0]])
        print sess.run(tf_cor_matrix_run(f, data))