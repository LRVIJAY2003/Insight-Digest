/* Main theme styles for Enterprise Knowledge Explorer */

:root {
    /* Light Theme Variables */
    --color-background: #FFFFFF;
    --color-sidebar: #F9FAFB;
    --color-text: #1F2937;
    --color-text-secondary: #6B7280;
    --color-primary: #3B82F6;
    --color-primary-light: #DBEAFE;
    --color-accent: #10B981;
    --color-border: #E5E7EB;
    --color-card: #FFFFFF;
    --color-card-hover: #F9FAFB;
    --color-error: #EF4444;
    --color-warning: #F59E0B;
    --color-success: #10B981;
    --color-info: #3B82F6;
    --color-code-bg: #F3F4F6;
    --color-mark: rgba(250, 204, 21, 0.35);
    --color-shadow: rgba(0, 0, 0, 0.1);
    --color-input-background: #F9FAFB;
    --color-input-border: #E5E7EB;
    --color-input-text: #1F2937;
}

/* Global Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    line-height: 1.5;
    color: var(--color-text);
    background-color: var(--color-background);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* App Layout */
.app-container {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    max-width: 1440px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Header Styles */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid var(--color-border);
    margin-bottom: 2rem;
}

.logo-container {
    display: flex;
    align-items: center;
}

.logo {
    color: var(--color-primary);
    margin-right: 1rem;
}

.app-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--color-text);
}

.header-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.theme-toggle {
    background: none;
    border: none;
    color: var(--color-text-secondary);
    cursor: pointer;
    font-size: 1.25rem;
    padding: 0.5rem;
    border-radius: 50%;
    transition: background-color 0.2s;
}

.theme-toggle:hover {
    background-color: var(--color-primary-light);
    color: var(--color-primary);
}

.theme-toggle .fa-sun {
    display: none;
}

.dark-mode .theme-toggle .fa-moon {
    display: none;
}

.dark-mode .theme-toggle .fa-sun {
    display: inline;
}

/* Main Content Area */
.app-main {
    flex: 1;
    padding-bottom: 2rem;
}

/* Search Section */
.search-section {
    margin-bottom: 2rem;
}

.search-container {
    max-width: 800px;
    margin: 0 auto;
}

.search-input-container {
    display: flex;
    position: relative;
    box-shadow: 0 4px 12px var(--color-shadow);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.search-input-container:focus-within {
    box-shadow: 0 8px 20px var(--color-shadow);
    transform: translateY(-2px);
}

.search-input-container input {
    flex: 1;
    border: none;
    padding: 1rem 1.5rem;
    font-size: 1.1rem;
    background-color: var(--color-input-background);
    color: var(--color-input-text);
    outline: none;
    border: 1px solid var(--color-input-border);
    border-right: none;
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
}

.search-button {
    background-color: var(--color-primary);
    color: white;
    border: none;
    padding: 0 1.5rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 120px;
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
}

.search-button:hover {
    background-color: #2563EB;
}

.search-options {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
}

.source-option {
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
}

.source-option input {
    position: absolute;
    opacity: 0;
    cursor: pointer;
    height: 0;
    width: 0;
}

.checkmark {
    position: relative;
    display: inline-block;
    height: 20px;
    width: 20px;
    background-color: var(--color-input-background);
    border: 1px solid var(--color-input-border);
    border-radius: 4px;
    margin-right: 8px;
}

.source-option:hover .checkmark {
    background-color: #EAEAEA;
}

.source-option input:checked ~ .checkmark {
    background-color: var(--color-primary);
    border-color: var(--color-primary);
}

.checkmark:after {
    content: "";
    position: absolute;
    display: none;
}

.source-option input:checked ~ .checkmark:after {
    display: block;
}

.source-option .checkmark:after {
    left: 7px;
    top: 3px;
    width: 5px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

.source-label {
    font-size: 0.95rem;
    color: var(--color-text-secondary);
}

/* Welcome Section */
.welcome-section {
    text-align: center;
    margin: 2rem auto;
    max-width: 800px;
    padding: 2rem;
    background-color: var(--color-card);
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--color-shadow);
}

.welcome-container h2 {
    font-size: 1.75rem;
    margin-bottom: 1rem;
    color: var(--color-text);
}

.welcome-container p {
    font-size: 1.1rem;
    color: var(--color-text-secondary);
    margin-bottom: 2rem;
}

.example-queries {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    justify-content: center;
}

.example-query-column {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    flex: 1;
    min-width: 280px;
}

.example-query {
    background-color: var(--color-primary-light);
    padding: 1rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    cursor: pointer;
    transition: transform 0.2s, background-color 0.2s;
    text-align: left;
}

.example-query:hover {
    transform: translateY(-2px);
    background-color: rgba(59, 130, 246, 0.2);
}

.example-query i {
    color: var(--color-primary);
    font-size: 1.2rem;
    width: 24px;
    text-align: center;
}

.example-query span {
    font-weight: 500;
    color: var(--color-text);
}

/* Results Section */
.results-section {
    max-width: 1200px;
    margin: 0 auto;
}

.keywords-container {
    display: flex;
    align-items: center;
    margin-bottom: 1.5rem;
    padding: 0.75rem 1rem;
    background-color: var(--color-card);
    border-radius: 8px;
    box-shadow: 0 2px 8px var(--color-shadow);
}

.keyword-title {
    font-weight: 600;
    margin-right: 1rem;
    white-space: nowrap;
    color: var(--color-text-secondary);
}

.keywords-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.keyword {
    background-color: var(--color-primary-light);
    color: var(--color-primary);
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.9rem;
    font-weight: 500;
}

.results-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}

.result-box {
    background-color: var(--color-card);
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--color-shadow);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.result-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px var(--color-shadow);
}

.result-header {
    background-color: var(--color-primary);
    color: white;
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.confluence-result .result-header {
    background-color: #0052CC;
}

.remedy-result .result-header {
    background-color: #E01E5A;
}

.result-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
}

.result-content {
    padding: 1.5rem;
    min-height: 200px;
    max-height: 500px;
    overflow-y: auto;
}

.result-content p {
    margin-bottom: 1rem;
    line-height: 1.6;
}

.result-content p:last-child {
    margin-bottom: 0;
}

.result-content ul, .result-content ol {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.result-content li {
    margin-bottom: 0.5rem;
}

.result-content p.sources {
    margin-top: 2rem;
    font-size: 0.9rem;
    color: var(--color-text-secondary);
    border-top: 1px solid var(--color-border);
    padding-top: 1rem;
}

.no-results {
    color: var(--color-text-secondary);
    font-style: italic;
}

.footer-info {
    display: flex;
    justify-content: flex-end;
    color: var(--color-text-secondary);
    font-size: 0.9rem;
}

.processing-time {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Error Message */
.error-message {
    background-color: #FECACA;
    border: 1px solid #F87171;
    color: #991B1B;
    padding: 1rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2rem;
}

.error-message i {
    font-size: 1.25rem;
    color: #DC2626;
}

/* Footer */
.app-footer {
    padding: 1.5rem 0;
    text-align: center;
    border-top: 1px solid var(--color-border);
    color: var(--color-text-secondary);
    font-size: 0.9rem;
}

/* Loading Spinner */
.spinner {
    display: inline-block;
}

.hidden {
    display: none;
}

/* Highlighted Keywords */
mark {
    background-color: var(--color-mark);
    padding: 0.1em 0.1em;
    border-radius: 2px;
}

/* Media Queries for Responsiveness */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.25rem;
    }
    
    .search-input-container {
        flex-direction: column;
        box-shadow: none;
    }
    
    .search-input-container input {
        border: 1px solid var(--color-input-border);
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    
    .search-button {
        width: 100%;
        padding: 0.75rem;
        border-radius: 12px;
    }
    
    .results-container {
        grid-template-columns: 1fr;
    }
    
    .keywords-container {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .keyword-title {
        margin-bottom: 0.5rem;
    }
}

@media (max-width: 480px) {
    .example-query-column {
        min-width: 100%;
    }
    
    .search-options {
        flex-direction: column;
        gap: 0.75rem;
        align-items: flex-start;
        margin-left: 1rem;
    }
    
    .app-header {
        flex-direction: column;
        gap: 1rem;
        text-align: center;
    }
    
    .logo-container {
        flex-direction: column;
    }
}