### Complications-aware Dynamic Classifier Selection for Unplanned Readmission Risk Prediction in Patients with Cirrhosis

#### Overview

This repository contains supplementary materials for the implementation of our submission titled **"Complications-aware Dynamic Classifier Selection for Unplanned Readmission Risk Prediction in Patients with Cirrhosis."**

**Our approach aims to:**

1. **Generate and characterize patient subgroups** using clinically meaningful rules based on complication and comorbidity patterns.
2. **Design a tailored region of competence** by incorporating medical diagnoses, enabling optimized dynamic selection of patient-specific predictive models.
3. **Establish a paradigm shift** from conventional one-size-fits-all approaches to an adaptive methodology that accounts for heterogeneous patient profiles.

By integrating medical knowledge with dynamic ensemble selection, our method not only improves predictive performance but also provides interpretable insights into the rationale behind classifier selection for each patient. This framework is implemented using the META-DES algorithm, relying on the [DESlib library](https://github.com/scikit-learn-contrib/DESlib).

> **Note:** The modifications made are not compatible with existing DES models.

---

#### Modifications in `deslib\des\meta-des.py`

A new set of meta-features (`meta_feature_f6`) was concatenated with existing meta-features.

**Example function:**

```python
    def compute_meta_features(self, scores, idx_neighbors, idx_neighbors_op, meta_feature_f6):

        idx_neighbors = np.atleast_2d(idx_neighbors)

        print('compute meta features')

        idx_neighbors_op = np.atleast_2d(idx_neighbors_op)

        f1_all_classifiers = self.DSEL_processed_[idx_neighbors, :]
        f1_all_classifiers = f1_all_classifiers.swapaxes(1, 2)
        f1_all_classifiers = f1_all_classifiers.reshape(-1, self.k_)
        f2_all_classifiers = \
            self.dsel_scores_[idx_neighbors, :,
            self.DSEL_target_[idx_neighbors]]

        f2_all_classifiers = f2_all_classifiers.swapaxes(1, 2)

        f2_all_classifiers = f2_all_classifiers.reshape(-1, self.k_)

        f3_all_classifiers = np.mean(self.DSEL_processed_[idx_neighbors, :],
                                     axis=1).reshape(-1, 1)

        f4_all_classifiers = self.DSEL_processed_[idx_neighbors_op, :]
        f4_all_classifiers = f4_all_classifiers.swapaxes(1, 2)
        f4_all_classifiers = f4_all_classifiers.reshape(-1, self.Kp_)

        f5_all_classifiers = np.max(scores, axis=2).reshape(-1, 1)

        meta_feature_vectors = np.hstack(
            (f1_all_classifiers, f2_all_classifiers, f3_all_classifiers,
             f4_all_classifiers, f5_all_classifiers, meta_feature_f6))

        print('meta feature vector shape')
        print(meta_feature_vectors.shape)

        return meta_feature_vectors
```

**Classifier selection based on maximum competence:**

```python
    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)
        print('competences')
        #print(competences)
        print(competences.shape)

        for i in range(len(competences)):
            max_val = np.max(competences[i]) # get the maximum value of the competence score
            for j in range(len(competences[i])):
                competences[i][j] = (competences[i][j] >= max_val)
        print(competences)
        selected_classifiers = np.where(competences == 0, False, True)

        # For the rows that are all False (i.e., no base classifier was
        # selected, select all classifiers (all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers
```

**Competence estimation with meta-feature f6:**

```python
    def estimate_competence_from_proba(self, neighbors, probabilities, meta_feature_f6,
                                       distances=None):

        _, idx_neighbors_op = self._get_similar_out_profiles(probabilities)
        meta_feature_vectors = self.compute_meta_features(probabilities,
                                                          neighbors,
                                                          idx_neighbors_op, meta_feature_f6)

        # Digitize the data if a Multinomial NB is used as the meta-classifier
        if isinstance(self.meta_classifier_, MultinomialNB):
            meta_feature_vectors = np.digitize(meta_feature_vectors,
                                                np.linspace(0.1, 1, 10))

        # Get the probability for class 1 (Competent)
        competences = self.meta_classifier_.predict_proba(
            meta_feature_vectors)[:, 1]

        # Reshape the array from 1D [n_samples x n_classifiers]
        # to 2D [n_samples, n_classifiers]
        competences = competences.reshape(-1, self.n_classifiers_)
        print('competences')
        print(competences)

        return competences
```

---

#### Modifications in `deslib\base.py`

The new set of meta-features is now passed to the `predict()` and `predict_proba()` methods to enhance dynamic classifier selection.

```python
    def predict(self, X, meta_feature):

        X = self._check_predict(X)

        preds = np.empty(X.shape[0], dtype=np.intp)
        need_proba = self.needs_proba or self.voting == 'soft'

        base_preds, base_probas = self._preprocess_predictions(X, need_proba)
        # predict all agree
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)

        if ind_all_agree.size:
            preds[ind_all_agree] = base_preds[ind_all_agree, 0]
        # predict with IH
        if ind_disagreement.size:
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                X, ind_disagreement, preds, is_proba=False
            )
            # Predict with DS - Check if there are still samples to be labeled.
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                
                meta_feature = meta_feature.reindex()
                print(len(inds))
                row_list = []
                for i in inds:
                    for j in range(i*self.n_classifiers_, (i+1)*self.n_classifiers_):
                        row_list.append(j)
                X1 = meta_feature.iloc[row_list]

                preds_ds = self.classify_with_ds(sel_preds, sel_probas,
                                                 neighbors, distances,
                                                 DFP_mask, X1)
                preds[inds] = preds_ds

        return self.classes_.take(preds)

    def predict_proba(self, X, meta_feature):

        X = self._check_predict(X)

        self._check_predict_proba()
        probas = np.zeros((X.shape[0], self.n_classes_))
        base_preds, base_probas = self._preprocess_predictions(X, True)
        # predict all agree
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        if ind_all_agree.size:
            probas[ind_all_agree] = base_probas[ind_all_agree].mean(axis=1)
        # predict with IH
        if ind_disagreement.size:
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                    X, ind_disagreement, probas, is_proba=True)
            # Predict with DS - Check if there are still samples to be labeled.
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)


                meta_feature = meta_feature.reindex()
                print(len(inds))
                row_list = []
                for i in inds:
                    for j in range(i*self.n_classifiers_, (i+1)*self.n_classifiers_):
                        row_list.append(j)
                X1 = meta_feature.iloc[row_list]

                probas_ds = self.predict_proba_with_ds(sel_preds,
                                                       sel_probas,
                                                       neighbors, distances,
                                                       DFP_mask, X1)
                probas[inds] = probas_ds
        return probas
```
---

#### Modifications in `deslib\des\base.py`

The new meta-feature set is also passed to `classify_with_ds()` and `predict_proba_with_ds()`.
```python
    def classify_with_ds(self, predictions, probabilities=None,
                         competence_region=None, distances=None,
                         DFP_mask=None, meta_feature=None):

        probas = self.predict_proba_with_ds(predictions, probabilities,
                                            competence_region, distances,
                                            DFP_mask, meta_feature)
        return probas.argmax(axis=1)

    def predict_proba_with_ds(self, predictions, probabilities=None,
                              competence_region=None, distances=None,
                              DFP_mask=None, meta_feature=None):

        if self.needs_proba:
            competences = self.estimate_competence_from_proba(
                neighbors=competence_region,
                distances=distances,
                meta_feature=meta_feature,
                probabilities=probabilities)
        else:
            competences = self.estimate_competence(
                competence_region=competence_region,
                distances=distances,
                predictions=predictions)
        if self.DFP:
            # FIRE-DES pruning.
            competences = competences * DFP_mask

        if self.mode == "selection":
            predicted_proba = self._dynamic_selection(competences,
                                                      predictions,
                                                      probabilities)
        elif self.mode == "weighting":
            predicted_proba = self._dynamic_weighting(competences, predictions,
                                                      probabilities)
        else:
            predicted_proba = self._hybrid(competences, predictions,
                                           probabilities)

        return predicted_proba

```
---

#### Generating the Diagnosis-based Region of Competence and New Meta-Features

The diagnosis-based region of competence consists of an equal number of positive and negative samples, enabling balanced estimation of base classifier performance. The meta-features include the predictions of base classifiers for these neighbors.
```python
    if extract_meta_feature == True:
        for col in diagnosis_features:
            preds = []
            preds_test = []
            exp = setup(data=X_train, target=col, session_id=123, fix_imbalance=True)
            for model in ['ada','dt','et','gbc','knn','lda','lightgbm','lr','nb','qda','rf','xgboost']:
                clf = create_model(model, cross_validation=False)

                pred = predict_model(clf, data=X_train)
                preds.append(pred['prediction_label'])

                pred_test = predict_model(clf, data=X_test)
                preds_test.append(pred_test['prediction_label'])

            temp_array = np.array(preds)
            temp_array = temp_array.T
            temp_array = temp_array.reshape((temp_array.shape[0]*temp_array.shape[1]))
            pred_feature[col] = temp_array
    
    nbrs_pos = NearestNeighbors(n_neighbors=n_neigh, algorithm='ball_tree').fit(X_train_df_pos[diagnosis_features])
    nbrs_neg = NearestNeighbors(n_neighbors=n_neigh, algorithm='ball_tree').fit(X_train_df_neg[diagnosis_features])

    distances_pos, indices_pos = nbrs_pos.kneighbors(X_train[diagnosis_features])
    distances_neg, indices_neg = nbrs_neg.kneighbors(X_train[diagnosis_features])

    indices_pos_re = indices_pos.reshape((X_train.shape[0]*n_neigh))
    indices_neg_re = indices_neg.reshape((X_train.shape[0]*n_neigh))

    pred_neighbor_feature = pd.DataFrame()
    
    # postive neighbors
    X_train_eval = X_train_df.loc[indices_pos_re]

    pred_array = []
    for model in pool_classifiers:
        y_pred = model.predict(X_train_eval)
        y_pred_re = y_pred.reshape((X_train.shape[0], n_neigh))
        pred_sum_pos = np.array(y_pred_re).sum(axis=1)
        pred_array.append(pred_sum_pos)

    pred_array = np.array(pred_array)
    pred_array = pred_array.T
    pred_array = pred_array.reshape((pred_array.shape[0]*pred_array.shape[1]))
    pred_neighbor_feature['positive_neighbor'] = pred_array

    # negative neighbors
    X_train_eval = X_train_df.loc[indices_neg_re]

    pred_array = []
    for model in pool_classifiers:
        y_pred = model.predict(X_train_eval)
        y_pred_re = y_pred.reshape((X_train.shape[0], n_neigh))
        pred_sum_pos = np.array(y_pred_re).sum(axis=1)
        pred_array.append(pred_sum_pos)

    pred_array = np.array(pred_array)
    pred_array = pred_array.T
    pred_array = pred_array.reshape((pred_array.shape[0]*pred_array.shape[1]))
    pred_neighbor_feature['negative_neighbor'] = pred_array

    meta_feature = pd.concat([pred_feature, pred_neighbor_feature], axis=1)
```
---

#### Training Base Classifiers with PyCaret

**Training process:**

1. Load dataset.
2. Set up the PyCaret environment.
3. Tune models.
4. Save models that achieve accuracy above 0.85.

```python
    dataset_path = 'subsets/...'
    dataset_list = os.listdir(dataset_path)
    path = os.getcwd().replace('\\', '/') + '/model/.../'
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    to_dir = path
    for datafile in dataset_list:
        final_df = pd.read_csv(dataset_path + '/' + datafile, encoding='ANSI')
        exp = setup(data=final_df, target='label', session_id=123, fix_imbalance=True)
        for model in ['gbc','rf','ada','qda','lr','xgboost','et','lightgbm','knn','lda','dt','nb']:
            clf = create_model(model, cross_validation=True)
            tuned = tune_model(clf, optimize = 'Accuracy')
            if pull()['Accuracy'][0] > 0.85:
                save_model(tuned, to_dir + datafile + '_' + model, model_only=True)
```
---

#### Evaluating the Proposed Framework

**Example usage:**

```python
method = METADES(pool_classifiers, random_state=rng, k=k, Kp=Kp, DSEL_perc=0.5, voting='soft', knne=False,
                 mode='weighting', selection_threshold=0.9, DFP=False, with_IH=False, knn_metric=knn_metric,
                 meta_feature=meta_feature)
method.fit(X_train, y_train)
y_pred = method.predict(X_test, meta_feature_test)
score = accuracy_score(y_test, y_pred)
y_prob = method.predict_proba(X_test, meta_feature_test)[:, 1]
```
---
