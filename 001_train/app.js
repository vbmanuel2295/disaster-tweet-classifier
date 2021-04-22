/**
 * This module is used for training a disaster tweets language classifier.
 *
 * 1. It loads data from the dataset/sample.csv
 * 2. It Translates the numeric class IDs to string based ones.
 * 3. It freezes the model's weights inside the ./disaster-tweet-classifier.json
 */
const natural = require("natural");
const csv = require("csvtojson");

let getLabel = (n) => {
    let lookup = new Map();

    lookup.set("0", "Not disaster related");
    lookup.set("1", "Disaster related");

    if (lookup.has(n)) {
        return lookup.get(n);
    }

    return "Unknown";
};

let persistModel = (classifier) => {
    return new Promise((resolve, reject) => {
        classifier.save("./disaster-tweet-classifier.json", (err) => {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
};

let loadData = () => {
    return new Promise((resolve) => {
        csv()
            .fromFile("./dataset/train.csv")
            .then((json) => {
                resolve(json);
            });
    });
};

(async () => {
    try {
        let dataset = await loadData();
        let classifier = new natural.BayesClassifier();

        dataset.forEach((i) => {
            let str_class = getLabel(i.target);

            classifier.addDocument(i.text, str_class);
        });

        classifier.train();

        await persistModel(classifier);
    } catch (err) {
        console.log(err.message);
        console.log(err.stack);
    }
})();