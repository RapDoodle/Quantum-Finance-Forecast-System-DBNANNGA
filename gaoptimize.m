function gaoptimize(model, X, y)
    probs = model.predict(X);
    m = size(y, 2);
    J = (1/m) * sum(sum((y-probs).^2));
    model.fitness = -log(J) + 0.5;
end

