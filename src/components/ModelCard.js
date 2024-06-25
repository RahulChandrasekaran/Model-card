// src/components/ModelCard.js
import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ModelCard = ({ model }) => {
  const [content, setContent] = useState('');

  useEffect(() => {
    if (model) {
      fetch(model.filePath)
        .then(response => response.text())
        .then(text => setContent(text))
        .catch(error => console.error('Error loading markdown file:', error));
    }
  }, [model]);

  if (!model) return <div>Select a model to view its details</div>;

  return (
    <div className="model-card">
      <h2>{model.name}</h2>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default ModelCard;
