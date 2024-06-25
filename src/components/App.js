// src/components/App.js
import React, { useState, useEffect } from 'react';
import ModelCardList from './ModelCardList';
import ModelCard from './ModelCard';
import '../App.css';

const App = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    // Automatically import all markdown files in the modelCards directory
    const modelCardsContext = require.context('../../public/modelCards', false, /\.md$/);
    const loadedModels = modelCardsContext.keys().map(filePath => {
      const fileName = filePath.split('/').pop();
      const modelName = fileName.replace('.md', '');
      return { name: modelName, filePath: `/modelCards/${fileName}` };
    });
    setModels(loadedModels);
  }, []);

  const handleSearchChange = event => {
    setSearchTerm(event.target.value);
  };

  const filteredModels = models.filter(model =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="app">
      <div className="sidebar">
        <input
          type="text"
          placeholder="Search models..."
          value={searchTerm}
          onChange={handleSearchChange}
        />
        <ModelCardList models={filteredModels} onSelect={setSelectedModel} />
      </div>
      <div className="model-details">
        <ModelCard model={selectedModel} />
      </div>
    </div>
  );
};

export default App;
