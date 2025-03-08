import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";

import "./LoginPage.css";

const LoginPage: React.FC = () => {
  return (
    <Main disableHeaderOffset>
      <Logo hasText />
    </Main>
  );
};

export default LoginPage;
