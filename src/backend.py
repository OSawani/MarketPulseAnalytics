import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error,make_scorer,mean_absolute_percentage_error
from datetime import timedelta
import plotly.express as px
import plotly.io as pio


def extract_date_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def predict_stock_prices(processed_df_stock_prices, symbol, date_start, date_end,y_predicted_target_desired='Low'):
    colors_dark = dict(zip(['Open', 'High', 'Low', 'Close'], ['gray', 'magenta', 'darkblue', 'green']))
    colors_light = {
        'Open': 'lightgray',
        'High': 'lavenderblush',
        'Low': 'lightblue',
        'Close': 'lightgreen'
    }
    valid_targets = {
        'Low': ['Open', 'High', 'Close', 'Volume', 'year', 'month', 'day', 'day_of_week', 'is_weekend'],
        'High': ['Open', 'Low', 'Close', 'Volume', 'year', 'month', 'day', 'day_of_week', 'is_weekend'],
        'Open': ['High', 'Low', 'Close', 'Volume', 'year', 'month', 'day', 'day_of_week', 'is_weekend'],
        'Close': ['Open', 'High', 'Low', 'Volume', 'year', 'month', 'day', 'day_of_week', 'is_weekend'],
        'Volume': ['Open', 'High', 'Low', 'Close', 'year', 'month', 'day', 'day_of_week', 'is_weekend'#]
                 ,'Open_Lag1','Open_Lag3','Open_Lag7','Close_Lag1','Close_Lag3','Close_Lag7','High_Lag1',
                'High_Lag3','High_Lag7','Low_Lag1','Low_Lag3','Low_Lag7','Volume_Lag1','Volume_Lag3',
                'Volume_Lag7','Open_MA3','Open_MA7','Close_MA3','Close_MA7','High_MA3','High_MA7',
                'Low_MA3','Low_MA7','Volume_MA3','Volume_MA7'#]
                ,'insider_TransactionValue_MA7','insider_TRANS_PRICEPERSHARE_Lag7','insider_TRANS_SHARES_Lag7',
                'insider_TransactionValue_MA21','insider_TRANS_PRICEPERSHARE_Lag21','insider_TRANS_SHARES_Lag21',]
    }
    test_percentage_split=0.2
    # y_predicted_target_desired = 'Low' # can be any of the 4 targets: 'Low', 'High', 'Open', 'Close'
    metrics = {}
    # Create subplots layout
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Stock Prices Predictions', y_predicted_target_desired+' vs Delta Volume Predictions/Actual','Features Importance for Volume Prediction'))
    fig.update_xaxes(showticklabels=True, row=1, col=1)

    dictionary_of_low_high_open_close_df_results = {'Low':None, 'High':None, 'Open':None, 'Close':None}
    


    # Process 'Low', 'High', 'Open', 'Close'
    for target in ['Low', 'High', 'Open', 'Close']:
        numerical_features = valid_targets[target]
        categorical_features = ['SYMBOL', 'Exists in Insiders']
        
        data = processed_df_stock_prices[
            (processed_df_stock_prices['SYMBOL'] == symbol) &
            (processed_df_stock_prices['Date'] >= date_start) &
            (processed_df_stock_prices['Date'] <= date_end)
        ].copy()
        data = extract_date_features(data)
        
        X = data.drop(target, axis=1)
        y = data[target]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        
 
        data = data.sort_values(by='Date')

        split_index = int(len(data) * (1 - test_percentage_split))

        # Custom split for training and testing based on sorted date
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage_split, random_state=0)
        # print("X_test Exists in Insiders count: ", X_test['Exists in Insiders'].value_counts())
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        root_mse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        metrics[target] = {'R²': r2, 'RMSE': root_mse}
        # to visualize train and test data we will force X_test to be the concatenation of X_train and X_test
        X_test = pd.concat([X_train, X_test])
        y_test = pd.concat([y_train, y_test])
        predictions = np.concatenate([pipeline.predict(X_train), predictions])
        

        df_results = pd.DataFrame({
            'Date': X_test['year'].astype(str) + '-' + 
                    X_test['month'].astype(str).str.zfill(2) + '-' + 
                    X_test['day'].astype(str).str.zfill(2),
            'Actual': y_test,
            'Predicted': predictions,
            'Exists in Insiders': X_test['Exists in Insiders']
        }).reset_index(drop=True)
        df_results['Date'] = pd.to_datetime(df_results['Date'])
        df_results.sort_values('Date', inplace=True)
        fig.add_trace(
            go.Scatter(
                x=df_results['Date'],
                y=df_results['Actual'],
                mode='lines',
                name=f'Actual {target}',
                line=dict(color=colors_light[target], width=2),
                hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Actual {target}: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_results['Date'],
                y=df_results['Predicted'],
                mode='lines',
                name=f'Predicted {target}',
                line=dict(color=colors_dark[target], width=2),
                hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Predicted {target}: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )
        # add df_results to dictionary
        dictionary_of_low_high_open_close_df_results[target] = df_results
    
    annotation_text = " | ".join([f"{target} - R²: {metrics[target]['R²']:.2f}, RMSE: {metrics[target]['RMSE']:.2f}" for target in ['Low', 'High', 'Open', 'Close']])
    




    # Add the 'Volume' subplot
    data = processed_df_stock_prices[
            (processed_df_stock_prices['SYMBOL'] == symbol) &
            (processed_df_stock_prices['Date'] >= date_start) &
            (processed_df_stock_prices['Date'] <= date_end)
        ].copy()
    data = extract_date_features(data)
    target = 'Volume'
    numerical_features = valid_targets[target]
    categorical_features = ['SYMBOL','Exists in Insiders','InsiderTransactionInLast7Days','InsiderTransactionInLast21Days']


    X = data.drop(target, axis=1)
    y = data[target]
    preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
    
    data = data.sort_values(by='Date')
    split_index = int(len(data) * (1 - test_percentage_split))

    # Custom split for training and testing based on sorted date
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Best hyperparameters
    best_params = {
        'regressor__colsample_bytree': 0.8,
        'regressor__learning_rate': 0.05,
        'regressor__max_depth': 4,
        'regressor__n_estimators': 300,
        'regressor__subsample': 0.9
    }

    # Update model with best hyperparameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=0,
        colsample_bytree=best_params['regressor__colsample_bytree'],
        learning_rate=best_params['regressor__learning_rate'],
        max_depth=best_params['regressor__max_depth'],
        n_estimators=best_params['regressor__n_estimators'],
        subsample=best_params['regressor__subsample']
    )
    pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage_split, random_state=0)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    root_mse = root_mean_squared_error(y_test, predictions)
    mean_absolute_percentage_error_scorer = mean_absolute_percentage_error(y_test,predictions)
    r2 = r2_score(y_test, predictions)
    metrics[target] = {'R²': r2, 'RMSE': root_mse, 'MAPE': mean_absolute_percentage_error_scorer}
    # to visualize training and testing data, we now force X_test to be the concatenation of the actual X_train and X_test
    X_test = pd.concat([X_train,X_test])
    #  thing for y_test
    y_test = pd.concat([y_train,y_test])
    # same thing for predictions
    predictions = np.concatenate([pipeline.predict(X_train),predictions])

    df_results = pd.DataFrame({
        'Date': X_test['year'].astype(str) + '-' + 
                X_test['month'].astype(str).str.zfill(2) + '-' + 
                X_test['day'].astype(str).str.zfill(2),
        'Actual': y_test,
        'Predicted': predictions,
        'Exists in Insiders': X_test['Exists in Insiders']
    }).reset_index(drop=True)
    df_results['Date'] = pd.to_datetime(df_results['Date'])
    # print("X_test Exists in Insiders count: ", X_test['Exists in Insiders'].value_counts())
    df_results['Delta'] = abs(df_results['Actual'] - df_results['Predicted'])
    df_results.sort_values('Date', inplace=True)
    
    
    scatter_fig = px.scatter(
        df_results,
        x='Date',
        y=dictionary_of_low_high_open_close_df_results[y_predicted_target_desired]['Predicted'],
        size='Delta',
        hover_data={'Date': '|%Y-%m-%d', 'Actual': ':.2e', 'Predicted': ':.2e', 'Delta': ':.2e'},
        color='Exists in Insiders',
        color_discrete_map={True: 'green', False: 'red'},
        labels={'color': 'Insider Status'},
        category_orders={'Exists in Insiders': [True, False]},
    )
    scatter_fig.for_each_trace(lambda t: t.update(name=t.name.replace("True", "Exists in Insiders").replace("False", "Does not exist in Insiders")))
    # Extract the trace and add it to the existing figure
    for trace in scatter_fig.data:
        trace.hovertemplate = (
            f'Date: %{{x|%Y-%m-%d}}<br>'
            f'predicted_{y_predicted_target_desired} = %{{y:.2f}}<br>'
            f'Delta: %{{marker.size:.2e}}<extra></extra>'  # Ensure Delta is included in the hovertemplate
        )
        fig.add_trace(trace, row=2, col=1)
    
    # the third row of the subplot will be bars plot of the features importance used in predicting 'volume'
    # get the feature importance from the pipeline
    




    # Get the numerical and categorical feature names after preprocessing
    numerical_features_out = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features)
    categorical_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

    # Combine the feature names
    all_feature_names = list(numerical_features_out) + list(categorical_features_out)

    # Check if the length matches feature importances
    if len(all_feature_names) != len(pipeline.named_steps['regressor'].feature_importances_):
        print("Mismatch detected: Adjusting feature names based on model output length.")
        # Adjust the feature names to match the length of feature importances
        all_feature_names = all_feature_names[:len(pipeline.named_steps['regressor'].feature_importances_)]

    # Plot feature importance
    fig.add_trace(
        go.Bar(
            x=all_feature_names,
            y=pipeline.named_steps['regressor'].feature_importances_,
            marker_color='lightblue'
        ),
        row=3, col=1
    )






    
    annotation_text_volume_only = f"{y_predicted_target_desired} vs Volume - R²: {metrics['Volume']['R²']:.2e}, RMSE: {metrics['Volume']['RMSE']:.2e}, MAPE: {metrics['Volume']['MAPE']:.2e}"
    # let's calculate the average delta for all points when Exists in Insiders is True
    average_delta_true = df_results[df_results['Exists in Insiders'] == True]['Delta'].mean()
    # let's calculate the average delta for all points when Exists in Insiders is False
    average_delta_false = df_results[df_results['Exists in Insiders'] == False]['Delta'].mean()
    # let's calculate the average delta for all points
    average_delta = df_results['Delta'].mean()
    # before conccaenating the average delta to the annotation text, let's concat a new line break using <br>
    annotation_text_volume_only += "<br>"
    annotation_text_volume_only += f" Avg. Δ (With Insid.): {average_delta_true:.2e}"
    # annotation_text_volume_only += "<br>"
    annotation_text_volume_only += f" | Avg. Δ (No Insid.): {average_delta_false:.2e}"
    # annotation_text_volume_only += "<br>"
    annotation_text_volume_only += f" | Avg. Δ: {average_delta:.2e}"
    
    # Add annotations for the first and second subplots
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.5, y=1.08,  # Position near the title of the first subplot
        xanchor='center',
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        text=annotation_text_volume_only,
        xref="paper", yref="paper",
        x=0.5, y=0.01,  # Position near the title of the second subplot
        xanchor='center',
        showarrow=False,
        font=dict(size=12)
    )

    # Layout and annotations
    fig.update_layout(
        # xaxis_title='Date',
        # yaxis_title='Value',
        height=800,
        # legend=dict(itemsizing='constant'),
        # hovermode='x unified'
    )
    # fig.show()
    
    return fig