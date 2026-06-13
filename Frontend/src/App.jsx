import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { ArrowDown, Cpu, Hash, Layers, Brain, Award, ZoomIn, ZoomOut } from 'lucide-react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [visibleSteps, setVisibleSteps] = useState([])
  const [zoom, setZoom] = useState(1.0)
  
  // Animation indices for 3-token concurrency
  const [preprocIdx, setPreprocIdx] = useState(0)
  const [ngramsIdx, setNgramsIdx] = useState(0)
  const [tfidfIdx, setTfidfIdx] = useState(0)
  const [selectorIdx, setSelectorIdx] = useState(0)

  const timeoutsRef = useRef([])

  const clearTimeouts = () => {
    timeoutsRef.current.forEach(t => clearTimeout(t))
    timeoutsRef.current = []
  }

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => clearTimeouts()
  }, [])

  // Auto-scroll to active tokens being processed
  useEffect(() => {
    const activeEl = document.getElementById(`token-preproc-${preprocIdx}`)
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'center' })
    }
  }, [preprocIdx])

  useEffect(() => {
    const activeEl = document.getElementById(`token-ngrams-${ngramsIdx}`)
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'center' })
    }
  }, [ngramsIdx])

  useEffect(() => {
    const activeEl = document.getElementById(`token-tfidf-${tfidfIdx}`)
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'center' })
    }
  }, [tfidfIdx])

  useEffect(() => {
    const activeEl = document.getElementById(`token-selector-${selectorIdx}`)
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'center' })
    }
  }, [selectorIdx])

  // Dynamic step auto-scrolling
  useEffect(() => {
    if (visibleSteps.length > 0) {
      const latestStepIdx = visibleSteps[visibleSteps.length - 1]
      const elementId = latestStepIdx === 5 ? 'step-wrapper-5' : `step-wrapper-${latestStepIdx}`
      const element = document.getElementById(elementId)
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }, 80)
      }
    }
  }, [visibleSteps])

  // Timers to drive 3-token processing windows
  useEffect(() => {
    if (visibleSteps.includes(0) && data) {
      const words = data.steps[1].aligned_words.filter(w => w.status !== 'ignored')
      if (words.length === 0) {
        setVisibleSteps(prev => [...prev, 1])
        return
      }
      const interval = setInterval(() => {
        setPreprocIdx(prev => {
          if (prev < words.length) {
            return prev + 1
          } else {
            clearInterval(interval)
            // Trigger next step
            setVisibleSteps(p => p.includes(1) ? p : [...p, 1])
            return prev
          }
        })
      }, 25) // Fast processing pace
      return () => clearInterval(interval)
    }
  }, [visibleSteps, data])

  useEffect(() => {
    if (visibleSteps.includes(1) && data) {
      const ngrams = data.steps[2].content
      if (ngrams.length === 0) {
        setVisibleSteps(prev => [...prev, 2])
        return
      }
      const interval = setInterval(() => {
        setNgramsIdx(prev => {
          if (prev < ngrams.length) {
            return prev + 1
          } else {
            clearInterval(interval)
            setVisibleSteps(p => p.includes(2) ? p : [...p, 2])
            return prev
          }
        })
      }, 25)
      return () => clearInterval(interval)
    }
  }, [visibleSteps, data])

  useEffect(() => {
    if (visibleSteps.includes(2) && data) {
      const ngrams = data.steps[2].content
      if (ngrams.length === 0) {
        setVisibleSteps(prev => [...prev, 3])
        return
      }
      const interval = setInterval(() => {
        setTfidfIdx(prev => {
          if (prev < ngrams.length) {
            return prev + 1
          } else {
            clearInterval(interval)
            setVisibleSteps(p => p.includes(3) ? p : [...p, 3])
            return prev
          }
        })
      }, 25)
      return () => clearInterval(interval)
    }
  }, [visibleSteps, data])

  useEffect(() => {
    if (visibleSteps.includes(3) && data) {
      const tokenizingStep = data.steps.find(s => s.name === 'Tokenizing')
      const tokensList = tokenizingStep ? tokenizingStep.content : []
      if (tokensList.length === 0) {
        setVisibleSteps(prev => [...prev, 4])
        return
      }
      const interval = setInterval(() => {
        setSelectorIdx(prev => {
          if (prev < tokensList.length) {
            return prev + 1
          } else {
            clearInterval(interval)
            setVisibleSteps(p => p.includes(4) ? p : [...p, 4])
            // Trigger final hero box
            const tId = setTimeout(() => {
              setVisibleSteps(p => p.includes(5) ? p : [...p, 5])
            }, 1000)
            timeoutsRef.current.push(tId)
            return prev
          }
        })
      }, 25)
      return () => clearInterval(interval)
    }
  }, [visibleSteps, data])

  const handlePredict = async () => {
    if (!text.trim()) return
    setLoading(true)
    setData(null)
    setVisibleSteps([])
    setPreprocIdx(0)
    setNgramsIdx(0)
    setTfidfIdx(0)
    setSelectorIdx(0)
    clearTimeouts()

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', { text })
      setData(response.data)
      setVisibleSteps([0]) // Start preprocessing immediately
    } catch (err) {
      console.error(err)
      alert('Error fetching prediction')
    } finally {
      setLoading(false)
    }
  }

  const renderPreprocessing = (step) => {
    const words = step.aligned_words.filter(w => w.status !== 'ignored')
    
    return (
      <div className="diff-container" style={{ fontSize: `${0.85 * zoom}rem` }}>
        <div className="diff-legend" style={{ fontSize: `${0.75 * zoom}rem` }}>
          <div className="legend-item">
            <span className="legend-color kept" style={{ background: 'rgba(59, 130, 246, 0.12)', border: '1px solid rgba(59, 130, 246, 0.4)' }}></span>
            <span>Kept</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: 'rgba(124, 58, 237, 0.08)', border: '1px solid rgba(124, 58, 237, 0.3)' }}></span>
            <span>Replaced</span>
          </div>
          <div className="legend-item">
            <span className="legend-color removed"></span>
            <span>Removed</span>
          </div>
        </div>
        
        <div className="token-cloud" style={{ overflowX: 'auto', paddingBottom: '0.5rem' }}>
          {words.map((word, i) => {
            // Processing Queue States:
            // i < preprocIdx: Processed. Discarded ones are removed from render, kept are shown.
            // i >= preprocIdx && i < preprocIdx + 3: Active animation.
            // i >= preprocIdx + 3: Hidden.
            
            if (i >= preprocIdx + 3) {
              return null
            }
            if (i < preprocIdx) {
              if (word.status === 'removed') return null
              const bg = word.status === 'replaced' ? 'rgba(124, 58, 237, 0.08)' : 'rgba(59, 130, 246, 0.12)'
              const border = word.status === 'replaced' ? '1px solid rgba(124, 58, 237, 0.3)' : '1px solid rgba(59, 130, 246, 0.4)'
              const color = word.status === 'replaced' ? '#5b21b6' : '#1e3a8a'
              return (
                <span 
                  key={i} 
                  className="token-item" 
                  style={{ background: bg, border: border, color: color, fontSize: `${11 * zoom}px` }}
                >
                  {
                  word.status === 'replaced'
                    ? word.replacement
                    : word.processed || word.text
                }
                </span>
              )
            }
            
            // Animating items
            const isFirstActive = i === preprocIdx
            const delay = `${(i - preprocIdx) * 0.1}s`
            
            if (word.status === 'kept') {
              return (
                <span 
                  key={i} 
                  id={isFirstActive ? `token-preproc-${i}` : undefined}
                  className="token-item reveal-kept" 
                  style={{ 
                    background: 'rgba(59, 130, 246, 0.12)', 
                    border: '1px solid rgba(59, 130, 246, 0.4)', 
                    color: '#1e3a8a',
                    fontSize: `${11 * zoom}px`,
                    animationDelay: delay
                  }}
                >
                  {word.text}
                  <span className="tooltip">Lemma: "{word.processed}"</span>
                </span>
              )
            } else if (word.status === 'replaced') {
              return (
                <span 
                  key={i} 
                  id={isFirstActive ? `token-preproc-${i}` : undefined}
                  className="token-item reveal-kept" 
                  style={{ 
                    background: 'rgba(124, 58, 237, 0.08)', 
                    border: '1px solid rgba(124, 58, 237, 0.3)', 
                    color: '#5b21b6',
                    fontSize: `${11 * zoom}px`,
                    animationDelay: delay 
                  }}
                >
                  {word.text}
                  <span className="tooltip" style={{ color: '#7c3aed' }}>Replaced: {word.replacement} ({word.reason})</span>
                </span>
              )
            } else {
              return (
                <span 
                  key={i} 
                  id={isFirstActive ? `token-preproc-${i}` : undefined}
                  className="token-item collapse-discarded" 
                  style={{ 
                    background: 'rgba(220, 38, 38, 0.08)', 
                    border: '1px solid rgba(220, 38, 38, 0.3)', 
                    color: '#dc2626',
                    fontSize: `${11 * zoom}px`,
                    animationDelay: delay 
                  }}
                >
                  {word.text}
                  <span className="tooltip">Removed: {word.reason || 'filtered'}</span>
                </span>
              )
            }
          })}
        </div>
      </div>
    )
  }

  const renderGeneratingNgrams = (step) => {
    return (
      <div className="token-cloud" style={{ fontSize: `${0.85 * zoom}rem` }}>
        {step.content.map((item, i) => {
          if (i >= ngramsIdx + 3) return null
          
          const isFirstActive = i === ngramsIdx
          const delay = `${(i - ngramsIdx) * 0.1}s`
          
          return (
            <span 
              key={i} 
              id={isFirstActive ? `token-ngrams-${i}` : undefined}
              className="token-item reveal-kept" 
              style={{ 
                background: 'rgba(79, 70, 229, 0.08)', 
                border: '1px solid rgba(79, 70, 229, 0.3)', 
                color: '#4f46e5',
                fontSize: `${11 * zoom}px`,
                animationDelay: delay
              }}
            >
              {item.ngram}
            </span>
          )
        })}
      </div>
    )
  }

  const renderTfidfTransforming = (step) => {
    return (
      <div className="token-cloud" style={{ fontSize: `${0.85 * zoom}rem` }}>
        {step.content.map((item, i) => {
          if (i >= tfidfIdx + 3) return null
          
          if (i < tfidfIdx) {
            if (!item.in_vocab) return null
            return (
              <span 
                key={i} 
                className="token-item" 
                style={{ 
                  background: 'rgba(52, 211, 153, 0.12)', 
                  border: '1px solid rgba(52, 211, 153, 0.4)', 
                  color: '#065f46',
                  fontWeight: 700,
                  fontSize: `${11 * zoom}px` 
                }}
              >
                {item.ngram}
                {item.weight > 0 && <span className="token-weight-badge" style={{ color: '#047857' }}>TF-IDF: {item.weight.toFixed(3)}</span>}
              </span>
            )
          }

          const isFirstActive = i === tfidfIdx
          const className = item.in_vocab ? 'reveal-kept' : 'collapse-discarded'
          const bg = item.in_vocab ? 'rgba(52, 211, 153, 0.12)' : 'rgba(220, 38, 38, 0.08)'
          const border = item.in_vocab ? '1px solid rgba(52, 211, 153, 0.4)' : '1px solid rgba(220, 38, 38, 0.3)'
          const color = item.in_vocab ? '#065f46' : '#dc2626'
          const delay = `${(i - tfidfIdx) * 0.15}s`

          return (
            <span 
              key={i} 
              id={isFirstActive ? `token-tfidf-${i}` : undefined}
              className={`token-item ${className}`} 
              style={{ 
                background: bg, 
                border: border, 
                color: color,
                textDecoration: item.in_vocab ? 'none' : 'line-through',
                fontWeight: item.in_vocab ? 700 : 400,
                fontSize: `${11 * zoom}px`,
                animationDelay: delay
              }}
            >
              {item.ngram}
              {item.in_vocab && item.weight > 0 && (
                <span className="token-weight-badge" style={{ color: '#047857' }}>TF-IDF: {item.weight.toFixed(3)}</span>
              )}
            </span>
          )
        })}
      </div>
    )
  }

  const renderSelector = (step) => {
    const tokenizingStep = data.steps.find(s => s.name === 'Tokenizing')
    const tokensList = tokenizingStep ? tokenizingStep.content : []
    
    const selectorMap = {}
    step.content.forEach(f => {
      selectorMap[f.ngram] = f
    })

    return (
      <div className="token-cloud" style={{ fontSize: `${0.85 * zoom}rem` }}>
        {tokensList.map((item, i) => {
          if (i >= selectorIdx + 3) return null
          
          const matchedFeature = selectorMap[item.ngram]
          const isSelected = matchedFeature ? matchedFeature.selected : false
          const chiScore = matchedFeature ? matchedFeature.chi_score : 0.0

          if (i < selectorIdx) {
            if (!isSelected) return null
            return (
              <span 
                key={i} 
                className="token-item" 
                style={{ 
                  background: 'rgba(139, 92, 246, 0.12)', 
                  border: '1px solid rgba(139, 92, 246, 0.4)', 
                  color: '#5b21b6',
                  fontWeight: 700,
                  fontSize: `${11 * zoom}px` 
                }}
              >
                {item.ngram}
                <span className="token-weight-badge" style={{ color: '#6d28d9', background: 'rgba(139, 92, 246, 0.1)' }}>
                  Chi2: {chiScore.toFixed(1)}
                </span>
              </span>
            )
          }

          const isFirstActive = i === selectorIdx
          const className = isSelected ? 'reveal-kept' : 'collapse-discarded'
          const bg = isSelected ? 'rgba(139, 92, 246, 0.12)' : 'rgba(220, 38, 38, 0.08)'
          const border = isSelected ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid rgba(220, 38, 38, 0.3)'
          const color = isSelected ? '#5b21b6' : '#dc2626'
          const delay = `${(i - selectorIdx) * 0.15}s`
          
          return (
            <span 
              key={i} 
              id={isFirstActive ? `token-selector-${i}` : undefined}
              className={`token-item ${className}`} 
              style={{ 
                background: bg, 
                border: border, 
                color: color,
                textDecoration: isSelected ? 'none' : 'line-through',
                fontWeight: isSelected ? 700 : 400,
                fontSize: `${11 * zoom}px`,
                animationDelay: delay
              }}
            >
              {item.ngram}
              {isSelected && (
                <span className="token-weight-badge" style={{ color: '#6d28d9', background: 'rgba(139, 92, 246, 0.1)' }}>
                  Chi2: {chiScore.toFixed(1)}
                </span>
              )}
            </span>
          )
        })}
      </div>
    )
  }

  return (
    <div className="container">
      <h1>Topic Classification Pipeline</h1>
      
      <div className="input-section">
        <textarea 
          placeholder="Enter text to classify (e.g. 'Bitcoin and Ethereum prices are rising...')"
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ fontSize: `${0.9 * zoom}rem`, height: '90px' }}
        />
        <div className="controls-wrapper">
          <div className="zoom-controls">
            <ZoomOut size={14} style={{ cursor: 'pointer', color: '#475569' }} onClick={() => setZoom(prev => Math.max(0.7, prev - 0.1))} />
            <span className="zoom-label" style={{ fontSize: `${0.75 * zoom}rem` }}>Zoom</span>
            <input 
              type="range" 
              min="0.7" 
              max="1.5" 
              step="0.1" 
              value={zoom} 
              onChange={(e) => setZoom(parseFloat(e.target.value))} 
              className="zoom-slider"
            />
            <span className="zoom-value" style={{ fontSize: `${0.8 * zoom}rem` }}>{Math.round(zoom * 100)}%</span>
            <ZoomIn size={14} style={{ cursor: 'pointer', color: '#475569' }} onClick={() => setZoom(prev => Math.min(1.5, prev + 0.1))} />
          </div>
          <button onClick={handlePredict} disabled={loading} style={{ fontSize: `${0.85 * zoom}rem`, padding: '0.5rem 1.5rem' }}>
            {loading ? 'Processing...' : 'Run Pipeline'}
          </button>
        </div>
      </div>

      {data && (
        <div className="pipeline">
          {/* Step 0: Preprocessing Stage */}
          {visibleSteps.includes(0) && (
            <div id="step-wrapper-0" className="step-wrapper">
              <div className="step-box">
                <div className="step-header">
                  <span>Preprocessing Stage</span>
                  <Cpu size={16} />
                </div>
                <div className="step-content" style={{ padding: '0.75rem 1rem' }}>
                  {renderPreprocessing(data.steps[1])}
                </div>
              </div>
              {visibleSteps.includes(1) && (
                <div className="arrow-container">
                  <ArrowDown size={18} />
                </div>
              )}
            </div>
          )}

          {/* Step 1: Generating Unigrams and Bigrams */}
          {visibleSteps.includes(1) && (
            <div id="step-wrapper-1" className="step-wrapper">
              <div className="step-box">
                <div className="step-header">
                  <span>Generating Unigrams and Bigrams</span>
                  <Hash size={16} />
                </div>
                <div className="step-content" style={{ padding: '0.75rem 1rem' }}>
                  {renderGeneratingNgrams(data.steps[2])}
                </div>
              </div>
              {visibleSteps.includes(2) && (
                <div className="arrow-container">
                  <ArrowDown size={18} />
                </div>
              )}
            </div>
          )}

          {/* Step 2: TFIDF Transforming */}
          {visibleSteps.includes(2) && (
            <div id="step-wrapper-2" className="step-wrapper">
              <div className="step-box">
                <div className="step-header">
                  <span>TF-IDF Transforming</span>
                  <Layers size={16} />
                </div>
                <div className="step-content" style={{ padding: '0.75rem 1rem' }}>
                  {renderTfidfTransforming(data.steps[2])}
                </div>
              </div>
              {visibleSteps.includes(3) && (
                <div className="arrow-container">
                  <ArrowDown size={18} />
                </div>
              )}
            </div>
          )}

          {/* Step 3: Selecting from Chi-Square Selector */}
          {visibleSteps.includes(3) && (
            <div id="step-wrapper-3" className="step-wrapper">
              <div className="step-box">
                <div className="step-header">
                  <span>Selecting from Chi-Square Selector</span>
                  <Layers size={16} />
                </div>
                <div className="step-content" style={{ padding: '0.75rem 1rem' }}>
                  {renderSelector(data.steps[3])}
                </div>
              </div>
              {visibleSteps.includes(4) && (
                <div className="arrow-container">
                  <ArrowDown size={18} />
                </div>
              )}
            </div>
          )}

          {/* Step 4: Results (Confidence List) */}
          {visibleSteps.includes(4) && (
            <div id="step-wrapper-4" className="results-container" style={{ gap: '1rem' }}>
              <div className="results-box" style={{ padding: '1.25rem' }}>
                <div className="results-title" style={{ fontSize: `${1.1 * zoom}rem`, marginBottom: '1rem', paddingBottom: '0.5rem' }}>
                  <Brain size={18} />
                  <span>Classification Candidates (Relative Confidence Scores)</span>
                </div>
                <div className="labels-grid" style={{ gap: '0.5rem' }}>
                  {data.prediction.classes.map((cls, i) => {
                    const isWinner = cls.label === data.prediction.label
                    return (
                      <div 
                        key={i} 
                        className={`label-row ${isWinner ? 'winning-row' : ''}`}
                        style={{ fontSize: `${0.85 * zoom}rem`, padding: '0.6rem 1rem' }}
                      >
                        <span className="row-name">{cls.label}</span>
                        <div className="row-bar-container" style={{ height: '6px' }}>
                          <div 
                            className="row-bar-fill" 
                            style={{ 
                              width: `${(cls.confidence * 100).toFixed(1)}%`,
                              background: isWinner ? 'linear-gradient(to right, #4f46e5, #7c3aed)' : '#94a3b8'
                            }}
                          ></div>
                        </div>
                        <span className="row-val prob" style={{ color: isWinner ? '#7c3aed' : '#475569', fontWeight: 'bold' }}>
                          {(cls.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
              {visibleSteps.includes(5) && (
                <div className="arrow-container">
                  <Award size={18} style={{ color: '#7c3aed', margin: '0 auto' }} />
                </div>
              )}
            </div>
          )}

          {/* Step 5: Final Prediction Box */}
          {visibleSteps.includes(5) && (
            <div id="step-wrapper-5" className="final-prediction-box-container">
              <div className="final-prediction-box" style={{ padding: `${1.5 * zoom}rem`, borderRadius: '1rem', border: '1px solid rgba(124, 58, 237, 0.3)' }}>
                <div className="final-icon-wrapper" style={{ width: `${48 * zoom}px`, height: `${48 * zoom}px`, marginBottom: '0.75rem' }}>
                  <Award size={24 * zoom} />
                </div>
                <span className="final-header" style={{ fontSize: `${0.95 * zoom}rem`, marginBottom: '0.5rem' }}>Final Prediction</span>
                <span className="final-label" style={{ fontSize: `${2.4 * zoom}rem`, marginBottom: '0.75rem' }}>{data.prediction.label}</span>
                <div className="final-details" style={{ fontSize: `${0.85 * zoom}rem`, padding: '0.4rem 1.2rem' }}>
                  <span>Class ID: <strong>{data.prediction.class_id}</strong></span>
                  <span>Relative Confidence: <strong>{(data.prediction.score * 100).toFixed(2)}%</strong></span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
