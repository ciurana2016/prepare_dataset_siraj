def print_performance(model, test_data, test_labels):
    # Result
    corect_answers = 0

    for index, t in enumerate(test_data):
        prediction = model.predict([t])[0]
        expected = test_labels.values[index]
        if prediction == expected:
            corect_answers += 1

    print 'You got %s of %s correct answers' % (corect_answers, len(test_labels))