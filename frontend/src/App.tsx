
import './App.css'
import Dashboard from './components/dashboard'

function App() {

  return (
    <>
      <div className="min-h-screen bg-cover bg-center" style={{ backgroundImage: `url(${'/bg.jpg'})` }}>
        <Dashboard />
      </div>
    </>
  )
}
export default App
