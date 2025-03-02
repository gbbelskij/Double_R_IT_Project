import Logo from "../Logo/Logo";

import classes from "./Header.module.css";

const Header: React.FC = () => {
  return (
    <header className={classes.Header}>
      <Logo hasText />
      <button>Заменить</button>
    </header>
  );
};

export default Header;
