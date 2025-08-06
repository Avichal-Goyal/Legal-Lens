// import React, { useState } from 'react';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import { useAuth } from './context/AuthContext.jsx';
// import { Navbar } from './components/Navbar.jsx';
// import { AnalyzePage } from './pages/AnalyzePage.jsx';
// import { LoginPage } from './pages/LoginPage.jsx';
// import { SignUpPage } from './pages/SignUpPage.jsx';

// const ProtectedRoute = ({ children }) => {
//   const { user } = useAuth();
//   return user ? children : <Navigate to="/login" />;
// };

// function App() {
//   return (
//     <Router>
//       <div className="min-h-screen bg-slate-50">
//         <Navbar />
//         <Routes>
//           <Route
//             path="/"
//             element={
//               <ProtectedRoute>
//                 <AnalyzePage />
//               </ProtectedRoute>
//             }
//           />
//           <Route path="/login" element={<LoginPage />} />
//           <Route path="/signup" element={<SignUpPage />} />
//         </Routes>
//       </div>
//     </Router>
//   );
// }

// export default App;
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar.jsx';
import { AnalyzePage } from './pages/AnalyzePage.jsx';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-zinc-50">
        <Navbar />
        <Routes>
         
          <Route path="/" element={<AnalyzePage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;