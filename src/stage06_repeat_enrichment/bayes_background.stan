data {

  int<lower=1> R;
  int<lower=1> G;
  int<lower=1> S;
  int<lower=1> C;
  int<lower=1> M;
  array[R] int<lower=0> n;
  array[R] int<lower=0> k;
  array[R] int<lower=0, upper=C> class_index;
  array[R] int<lower=1, upper=G> genome;
  array[R] int<lower=0, upper=S> sample;
  vector[R] side;
  matrix[R, M] W;
}

parameters {

  real alpha;
  vector[C] delta;
  real beta;
  real<lower=0> sigma_genome;
  real<lower=0> sigma_sample;
  vector[G] z_genome;
  vector[S] z_sample;
}

transformed parameters {

  vector[R] eta0 = alpha + beta * side + sigma_genome * z_genome[genome];
  vector[R] eta = eta0;

  for (r in 1:R) {

    if (class_index[r] > 0) eta[r] += delta[class_index[r]] + sigma_sample * z_sample[sample[r]];
  }
}

model {

  vector[R] kk = to_vector(k);
  vector[R] nn = to_vector(n);

  target += dot_product(kk, log1m_exp(-exp(eta))) - dot_product(nn - kk, exp(eta));

  alpha ~ student_t(3, -3, 2.5);
  delta ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma_genome ~ exponential(log(20) / 5);
  sigma_sample ~ exponential(log(20) / 5);
  z_genome ~ std_normal();
  z_sample ~ std_normal();
}

generated quantities {

  vector[M] focal_rate = W' * (1 - exp(-exp(eta)));
  vector[M] background_rate = W' * (1 - exp(-exp(eta0)));
  vector[M] difference = focal_rate - background_rate;
  vector[M] ratio = focal_rate ./ background_rate;
}
