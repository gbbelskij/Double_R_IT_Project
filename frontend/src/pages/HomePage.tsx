import Footer from "../components/Footer/Footer";
import Header from "../components/Header/Header";
import Logo from "../components/Logo/Logo";
import Main from "../components/Main/Main";
import Input from "../components/Input/Input";

import "./HomePage.css";

export default function Home() {
  return (
    <>
      <Header />
      <Main>
        <Logo hasText />
        <div>
          <Input 
            type="number" 
            name="experience" 
            label="Опыт работы" 
            unit="лет"
          />
        </div>
      </Main>
    
      <Footer />
    </>
  );
}
