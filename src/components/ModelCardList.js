// src/components/ModelCardList.js
import React from 'react';

const ModelCardList = ({ models, onSelect }) => {
  return (
    <div className="model-card-list">
      <h2>Select a Model</h2>
      <ul>
        {models.map((model, index) => (
          <li key={index} onClick={() => onSelect(model)}>
            {model.name}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ModelCardList;
