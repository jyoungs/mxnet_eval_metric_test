## CODE SOURCE: https://github.com/apache/incubator-mxnet/issues/2138#issuecomment-269959977

require(mxnet)

## Translated to R from python example at:
# https://github.com/dmlc/mxnet/issues/2138#issuecomment-222812951

# MXNET settings:
nRounds <-300
nHidden <- 30
optimizer <- "rmsprop"
array.layout <- "rowmajor"
ctx <- mx.cpu()
initializer <- mx.init.Xavier()

# Data settings:
nObservations <- 2000
noiseLvl <- 0.5
nOutput <- 3
set.seed(42)
mx.set.seed(42)

get_mlp <- function() {
  # multi-layer perceptron
  label = mx.symbol.Variable('label')
  data = mx.symbol.Variable('data')
  flat = mx.symbol.Flatten(data=data)
  fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=nHidden)
  act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")
  fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=nOutput)
  net  = mx.symbol.LinearRegressionOutput(data=fc2, label=label, name='lro')
  return(net)
}

# Generate some random data
df <- data.frame(x1=rnorm(nObservations),
                 x2=rnorm(nObservations),
                 x3=rnorm(nObservations),
                 x4=rnorm(nObservations))
expts <- list()
for (outIdx in 1:nOutput) {
  expts[[outIdx]] <- sample(0:3, 4, replace=T)
  df[[paste0("y", outIdx)]] <- df$x1^expts[[outIdx]][1] +
    df$x2^expts[[outIdx]][2] + df$x3^expts[[outIdx]][3] +
    df$x4^expts[[outIdx]][4] + noiseLvl*rnorm(nObservations)
}

respCols <- paste0("y", 1:nOutput)

# Scale data to zero-mean unit-variance
df <- data.frame(scale(df))

# Split into training and test sets
test.ind = seq(1, nObservations, 10)  # 1 in 10 smaples for testing
train.x = data.matrix(df[-test.ind, -which(names(df) %in% respCols)])
train.y = data.matrix(df[-test.ind, respCols])
test.x = data.matrix(df[test.ind, -which(names(df) %in% respCols)])
test.y = data.matrix(df[test.ind, respCols])

# Setup iterators
trainIter = mx.io.arrayiter(data = t(train.x), label = t(train.y))
valIter   = mx.io.arrayiter(data = t(test.x) , label = t(test.y))

# Get model and train
net = get_mlp()

model = mx.model.FeedForward.create(X=trainIter,
                                    eval.data=valIter,
                                    ctx=ctx,
                                    symbol=net,
                                    num.round=nRounds,
                                    initializer=initializer,
                                    optimizer=optimizer,
                                    array.layout=array.layout
                                    )

# Prediction
train.Response <- t(predict(model, train.x, array.layout=array.layout))
test.Response <- t(predict(model, test.x, array.layout=array.layout))

# Results
print("Train rmse:")
print(colMeans((train.Response - train.y)^2))

print("Test rmse:")
print(colMeans((test.Response - test.y)^2))


par(mfrow=c(nOutput, 2))
for (outIdx in 1:nOutput) {
  plot(train.y[, outIdx], train.Response[, outIdx],
       xlab="Actual output", ylab="Model Response",
       main=paste0("train perf. output ", outIdx))
  abline(0,1)

  plot(test.y[, outIdx], test.Response[, outIdx],
       xlab="Actual output", ylab="Model Response",
       main=paste0("test perf output ", outIdx))
  abline(0,1)
}
