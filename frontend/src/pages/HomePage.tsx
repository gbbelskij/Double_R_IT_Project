import Footer from "../components/Footer/Footer";
import Header from "../components/Header/Header";
import Logo from "../components/Logo/Logo";
import Main from "../components/Main/Main";
import Button, { ButtonTypes } from "../components/Button/Button.tsx";
import { FaPaperPlane } from "react-icons/fa";

import "./HomePage.css";

export default function Home() {
  return (
    <>
      <Header />
      <Main>
        <Logo hasText />
        <div style={{ marginTop: "20px" }}>
          <h2>Button Examples</h2>
          <Button type={ButtonTypes.Action} leftIcon={<FaPaperPlane />}>
            Применить изменения
          </Button>
          <br /><br />
          <Button type={ButtonTypes.DangerousAction} fullWidth>
            Удалить аккаунт
          </Button>
          <br /><br />
          <Button type={ButtonTypes.Default}>Выйти из аккаунта</Button>
          <br /><br />
          <Button type={ButtonTypes.Submit} fullWidth>
            Завершить регистрацию
          </Button>
        </div>
      </Main>
      <Footer />
    </>
  );
}