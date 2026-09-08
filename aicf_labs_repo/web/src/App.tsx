import { useState } from 'react';
import type { Report } from './report';
import './style.css';

export default function App({ report }: { report: Report }) {
  const [view, setView] = useState<'original' | 'transformed'>('transformed');
  const [selectedId, setSelectedId] = useState('');
  const graph = report[view];
  const selected = graph.nodes.find((node) => node.id === selectedId) ?? graph.nodes[0];
  const implementation = view === 'transformed' ? report.cuda.find((item) => item.node_id === selected?.id) : undefined;
  return <main>
    <header><p>AICF · MINIMAL BACKEND VIEWER</p><h1>모델에서 CUDA 연결까지</h1>
      <span>Python backend가 생성한 결과 · 실행: {report.execution.backend}</span></header>
    <section aria-label="Model"><h2>Model</h2><div className="layers">{report.model.map((layer, index) =>
      <span key={index}>{index > 0 && '→ '}{layer.type}{layer.weight_shape && ` [${layer.weight_shape.join(' × ')}]`}</span>)}</div></section>
    <section><h2>Backend transformation</h2>{report.decisions.map((decision, index) =>
      <p key={index}><strong>{decision.rule} · {decision.status}</strong> — {decision.reason}
        <small>{decision.nodes.join(' → ')} · domain: {decision.domain}</small></p>)}</section>
    <div className="columns">
      <section><div className="graph-header"><h2>Graph</h2><div>
        <button aria-pressed={view === 'original'} onClick={() => setView('original')}>원본</button>
        <button aria-pressed={view === 'transformed'} onClick={() => setView('transformed')}>변환 후</button>
      </div></div><p>Output: <code>{graph.output}</code></p>
        <table><thead><tr><th>Operator instance</th><th>Input edges</th></tr></thead><tbody>{graph.nodes.map((node) =>
          <tr key={node.id}><td><button className="node" aria-pressed={selected?.id === node.id} onClick={() => setSelectedId(node.id)}>{node.id}</button></td>
            <td>{node.inputs.length ? node.inputs.map((input) => <small key={input}>{input} → {node.id}</small>) : '—'}</td></tr>)}</tbody></table>
      </section>
      {selected && <section aria-label="Selected operator"><h2>{selected.type}</h2>
        <p>Shape: [{selected.shape.join(', ')}]</p><code className="expression">{selected.semantics.expression}</code>
        <h3>Semantic properties</h3><ul>{selected.semantics.properties.map((property, index) =>
          <li key={index}>{property.kind}{property.argument !== null && ` (argument=${property.argument})`}
            {property.condition && <small>{property.condition}</small>}</li>)}</ul>
        <h3>Attributes</h3><pre>{JSON.stringify(selected.attributes, null, 2)}</pre>
        {implementation && <><h3>Selected CUDA implementation</h3><code>{implementation.implementation_id}</code>
          <p>CUDA launch: {implementation.launch.status} · 실제 GPU 실행 없음</p></>}
      </section>}
    </div>
    <section><h2>Reference verification</h2><p className={report.execution.equal ? 'pass' : 'fail'}>
      {report.execution.equal ? '원본과 변환 결과 일치' : '원본과 변환 결과 불일치'}</p>
      <div className="outputs"><div>Original<pre>{JSON.stringify(report.execution.original_output)}</pre></div>
        <div>Transformed<pre>{JSON.stringify(report.execution.transformed_output)}</pre></div></div>
      <small>{report.execution.scope}</small></section>
    <footer>모델을 바꾸려면 demo.py를 수정한 뒤 npm run report를 실행하세요. 이 화면은 보고서를 읽으며 규칙을 정의하거나 실행하지 않습니다.</footer>
  </main>;
}
